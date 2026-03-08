# ==================== IMPORTS ====================
import os, sys, glob, math, random, functools, zipfile
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

# ==================== PATCH torch.load FOR PyTorch 2.6+ ====================
_orig_torch_load = torch.load
torch.load = functools.partial(_orig_torch_load, weights_only=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==================== GLOBAL OUTPUT DIRECTORY ====================
OUT_DIR = None

def get_out_dir():
    global OUT_DIR
    if OUT_DIR is None:
        raise RuntimeError("OUT_DIR not set. Call set_out_dir() first.")
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR

def set_out_dir(data_root):
    global OUT_DIR
    OUT_DIR = os.path.join(data_root, "aux results v2")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  Output directory: {OUT_DIR}")
    return OUT_DIR


###############################################################################
#                           1.  LOSS FUNCTIONS                                #
###############################################################################

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.w = window_size
        self.register_buffer("_gauss", self._make_kernel(window_size))

    @staticmethod
    def _make_kernel(ws, sigma=1.5):
        ax = torch.arange(ws, dtype=torch.float32) - ws // 2
        g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        k = g.unsqueeze(1) @ g.unsqueeze(0)
        return k.unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        C = pred.size(1)
        kernel = self._gauss.expand(C, 1, -1, -1).to(pred.device)
        pad = self.w // 2
        mu_p = F.conv2d(pred,   kernel, padding=pad, groups=C)
        mu_t = F.conv2d(target, kernel, padding=pad, groups=C)
        s_pp = F.conv2d(pred * pred,     kernel, padding=pad, groups=C) - mu_p * mu_p
        s_tt = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_t * mu_t
        s_pt = F.conv2d(pred * target,   kernel, padding=pad, groups=C) - mu_p * mu_t
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * s_pt + C2)) / \
                   ((mu_p ** 2 + mu_t ** 2 + C1) * (s_pp + s_tt + C2))
        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg = models.vgg19(pretrained=True).features
        self.s1 = nn.Sequential(*list(vgg)[:4]).eval()
        self.s2 = nn.Sequential(*list(vgg)[4:9]).eval()
        self.s3 = nn.Sequential(*list(vgg)[9:18]).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        return super().train(False)

    def _norm(self, x):
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        p, t = self._norm(pred), self._norm(target)
        f1p, f1t = self.s1(p), self.s1(t)
        f2p, f2t = self.s2(f1p), self.s2(f1t)
        f3p, f3t = self.s3(f2p), self.s3(f2t)
        return F.l1_loss(f1p, f1t) + F.l1_loss(f2p, f2t) + F.l1_loss(f3p, f3t)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sx", sx.view(1, 1, 3, 3))
        self.register_buffer("sy", sy.view(1, 1, 3, 3))

    def forward(self, pred, target):
        px = F.conv2d(pred, self.sx, padding=1)
        py = F.conv2d(pred, self.sy, padding=1)
        tx = F.conv2d(target, self.sx, padding=1)
        ty = F.conv2d(target, self.sy, padding=1)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)


class CombinedLoss(nn.Module):
    def __init__(self, w_char=1.0, w_ssim=0.5, w_perc=0.02, w_edge=0.1):
        super().__init__()
        self.w = dict(char=w_char, ssim=w_ssim, perc=w_perc, edge=w_edge)
        self.char = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.perc = PerceptualLoss()
        self.edge = EdgeLoss()

    def forward(self, pred, target):
        L  = self.w["char"] * self.char(pred, target)
        L += self.w["ssim"] * self.ssim(pred, target)
        L += self.w["perc"] * self.perc(pred, target)
        L += self.w["edge"] * self.edge(pred, target)
        return L


###############################################################################
#                        2.  MODEL BUILDING BLOCKS                            #
###############################################################################

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid, ch, 1, bias=False)
        )

    def forward(self, x):
        a = self.fc(F.adaptive_avg_pool2d(x, 1))
        m = self.fc(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(a + m)


class SpatialAttention(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=ks // 2, bias=False)

    def forward(self, x):
        desc = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(desc))


class RCAB(nn.Module):
    def __init__(self, nf, r=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
        self.ca = ChannelAttention(nf, r)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x + self.sa(self.ca(self.body(x)))


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.c1 = nn.Conv2d(nf,         gc, 3, 1, 1)
        self.c2 = nn.Conv2d(nf + gc,    gc, 3, 1, 1)
        self.c3 = nn.Conv2d(nf + 2*gc,  gc, 3, 1, 1)
        self.c4 = nn.Conv2d(nf + 3*gc,  gc, 3, 1, 1)
        self.c5 = nn.Conv2d(nf + 4*gc,  nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.b1 = ResidualDenseBlock(nf, gc)
        self.b2 = ResidualDenseBlock(nf, gc)
        self.b3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.b3(self.b2(self.b1(x))) * 0.2 + x


###############################################################################
#         ★★★  KEY CHANGE: GUIDE-INJECTED RRDB BLOCKS  ★★★                    #
###############################################################################

class GuidedRRDB(nn.Module):
    """
    RRDB block that receives guide conditioning.
    Guide features modulate the output via learned gating.
    """
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rrdb = RRDB(nf, gc)
        # Lightweight guide projection
        self.guide_proj = nn.Sequential(
            nn.Conv2d(nf, nf, 1),
            nn.LeakyReLU(0.2, True)
        )
        # Gating mechanism: decides how much guide info to incorporate
        self.gate = nn.Sequential(
            nn.Conv2d(nf * 2, nf, 1),
            nn.Sigmoid()
        )

    def forward(self, x, guide_feat):
        """
        x:          thermal features (B, nf, H, W)
        guide_feat: guide features at same resolution (B, nf, H, W)
        """
        out = self.rrdb(x)
        g = self.guide_proj(guide_feat)
        gate = self.gate(torch.cat([out, g], 1))
        # Gated residual: blend RRDB output with skip based on guide
        return out * gate + x * (1 - gate)


class GuidedTrunk(nn.Module):
    """
    RRDB trunk with guide injection at EVERY block.
    This is the key architectural improvement.
    """
    def __init__(self, n_rrdb, nf=64, gc=32):
        super().__init__()
        self.blocks = nn.ModuleList([GuidedRRDB(nf, gc) for _ in range(n_rrdb)])
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x, guide_feat):
        h = x
        for block in self.blocks:
            h = block(h, guide_feat)
        return self.conv(h) + x   # long skip connection


###############################################################################
#                    3.  ENHANCED SFT FUSION                                  #
###############################################################################

class EnhancedSFTFusion(nn.Module):
    """SFT modulation + concatenation-attention path."""
    def __init__(self, nf):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.scale_net = nn.Sequential(
            nn.Conv2d(nf * 2, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.Sigmoid()
        )
        self.shift_net = nn.Sequential(
            nn.Conv2d(nf * 2, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
        self.ca = ChannelAttention(nf * 2)
        self.reduce = nn.Conv2d(nf * 2, nf, 1)

    def forward(self, thermal, guide):
        g = self.cond(guide)
        cat = torch.cat([thermal, g], 1)
        sft = thermal * self.scale_net(cat) + self.shift_net(cat)
        out = self.reduce(self.ca(torch.cat([sft, g], 1)))
        return out


###############################################################################
#                     4.  GUIDE ENCODER (ResNet-34)                           #
###############################################################################

class GuideEncoder(nn.Module):
    """
    Multi-scale feature extractor from HR visible image.
    Returns features at scales: 1 (full), 2, 4, 8, 16.
    """
    def __init__(self, nf=64):
        super().__init__()
        self.shallow = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        try:
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except Exception:
            resnet = models.resnet34(pretrained=True)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool   = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Freeze early layers
        for p in list(self.layer0.parameters()) + list(self.layer1.parameters()):
            p.requires_grad = False

        # Projections to nf channels
        self.p2  = nn.Conv2d(64,  nf, 1)
        self.p4  = nn.Conv2d(64,  nf, 1)
        self.p8  = nn.Conv2d(128, nf, 1)
        self.p16 = nn.Conv2d(256, nf, 1)

    def forward(self, vis):
        f0 = self.layer0(vis)
        f1 = self.layer1(self.pool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return {
            "s1":  self.shallow(vis),
            "s2":  self.p2(f0),
            "s4":  self.p4(f1),
            "s8":  self.p8(f2),
            "s16": self.p16(f3),
        }


###############################################################################
#                      5.  MAIN SR NETWORK                                    #
###############################################################################

class GuidedThermalSRNet(nn.Module):
    """
    LR thermal (1ch) + HR visible (3ch) → HR thermal (1ch)

    Architecture v2:
    ─────────────────
    1. Shallow head → Guide-injected RRDB trunk → long skip
    2. Guide features injected at EVERY trunk block (not just pre/post)
    3. Progressive ×2 pixel-shuffle stages with SFT fusion
    4. Final refinement → output + bicubic residual
    """
    def __init__(self, scale, nf=64, n_rrdb=12, gc=32, n_rcab_stage=3, n_rcab_final=4):
        super().__init__()
        self.scale = scale

        # Thermal head
        self.head = nn.Sequential(
            nn.Conv2d(1, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )

        # ★ Guide-injected trunk (KEY CHANGE)
        self.guided_trunk = GuidedTrunk(n_rrdb, nf, gc)

        # Guide encoder
        self.guide_enc = GuideEncoder(nf)

        # Pre-upsample fusion
        self.pre_fuse = EnhancedSFTFusion(nf)

        # Progressive upsample stages
        n_stages = int(math.log2(scale))
        self.n_stages = n_stages

        self.up_convs  = nn.ModuleList()
        self.fusions   = nn.ModuleList()
        self.refiners  = nn.ModuleList()

        # Determine guide key for each upsample stage
        self._guide_keys = []
        rem = scale
        for _ in range(n_stages):
            rem //= 2
            if   rem >= 16: self._guide_keys.append("s16")
            elif rem >= 8:  self._guide_keys.append("s8")
            elif rem >= 4:  self._guide_keys.append("s4")
            elif rem >= 2:  self._guide_keys.append("s2")
            else:           self._guide_keys.append("s1")

        for _ in range(n_stages):
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, True)
            ))
            self.fusions.append(EnhancedSFTFusion(nf))
            self.refiners.append(nn.Sequential(*[RCAB(nf) for _ in range(n_rcab_stage)]))

        # Final refinement
        self.final_ref = nn.Sequential(*[RCAB(nf) for _ in range(n_rcab_final)])
        self.out_conv  = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, 1, 3, 1, 1)
        )

    @property
    def _pre_key(self):
        s = self.scale
        if   s >= 16: return "s16"
        elif s >= 8:  return "s8"
        elif s >= 4:  return "s4"
        else:         return "s2"

    def forward(self, lr_therm, hr_vis):
        # Bicubic baseline for global residual
        bic = F.interpolate(lr_therm, scale_factor=self.scale,
                            mode="bicubic", align_corners=False)

        # Thermal features
        shallow = self.head(lr_therm)

        # Guide features (multi-scale)
        gf = self.guide_enc(hr_vis)

        # Get guide at LR resolution for trunk injection
        g_lr = F.interpolate(gf[self._pre_key], size=shallow.shape[2:],
                             mode="bilinear", align_corners=False)

        # ★ Guide-injected deep trunk (KEY CHANGE)
        deep = self.guided_trunk(shallow, g_lr)

        # Pre-upsample fusion
        x = self.pre_fuse(deep, g_lr)

        # Progressive upsample
        for i in range(self.n_stages):
            x = self.up_convs[i](x)
            g = F.interpolate(gf[self._guide_keys[i]], size=x.shape[2:],
                              mode="bilinear", align_corners=False)
            x = self.fusions[i](x, g)
            x = self.refiners[i](x)

        x = self.final_ref(x)
        out = self.out_conv(x)

        # Match spatial size exactly
        if out.shape[2:] != bic.shape[2:]:
            out = F.interpolate(out, size=bic.shape[2:],
                                mode="bilinear", align_corners=False)

        return out + bic   # Global residual


###############################################################################
#                            6.  EMA HELPER                                   #
###############################################################################

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


###############################################################################
#                        7.  DATASETS (CIDIS layout)                          #
###############################################################################

def _thermal_base_to_vis_base(thermal_base):
    """'477_01_D4_th' → '477_01_D4_vis'"""
    parts = thermal_base.rsplit("_th", 1)
    return "_vis".join(parts)


def _find_file(directory, base):
    for ext in (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = os.path.join(directory, base + ext)
        if os.path.exists(p):
            return p
    return None


class ThermalGuidedDataset(Dataset):
    """
    Train / Val dataset.
      {root}/thermal/{split}/LR_x{scale}/  *_th.bmp
      {root}/thermal/{split}/GT/            *_th.bmp
      {root}/visible/{split}/               *_vis.bmp
    """
    def __init__(self, root, scale, split="train", patch_size=256, augment=True):
        self.scale = scale
        self.ps    = patch_size
        self.aug   = augment and split == "train"
        self.split = split

        lr_dir  = os.path.join(root, "thermal", split, f"LR_x{scale}")
        hr_dir  = os.path.join(root, "thermal", split, "GT")
        vis_dir = os.path.join(root, "visible", split)

        for d, name in [(lr_dir, "LR"), (hr_dir, "GT"), (vis_dir, "Visible")]:
            if not os.path.isdir(d):
                print(f"  ⚠ WARNING: {name} directory not found: {d}")

        self.triplets = []
        for lr_path in sorted(glob.glob(os.path.join(lr_dir, "*.*"))):
            th_base  = os.path.splitext(os.path.basename(lr_path))[0]
            hr_path  = _find_file(hr_dir, th_base)
            vis_base = _thermal_base_to_vis_base(th_base)
            vis_path = _find_file(vis_dir, vis_base)
            if hr_path and vis_path:
                self.triplets.append((lr_path, hr_path, vis_path, th_base))

        print(f"  [{split:5s}] Loaded {len(self.triplets)} triplets  (scale ×{scale})")

    @staticmethod
    def _load_gray(path):
        return np.array(Image.open(path).convert("L"), np.float32) / 255.0

    @staticmethod
    def _load_rgb(path):
        return np.array(Image.open(path).convert("RGB"), np.float32) / 255.0

    def _random_crop(self, lr, hr, vis):
        lps = self.ps // self.scale
        lh, lw = lr.shape[:2]
        lps = min(lps, lh, lw)
        hps = lps * self.scale
        ly = random.randint(0, max(lh - lps, 0))
        lx = random.randint(0, max(lw - lps, 0))
        hy, hx = ly * self.scale, lx * self.scale
        return (lr[ly:ly+lps, lx:lx+lps],
                hr[hy:hy+hps, hx:hx+hps],
                vis[hy:hy+hps, hx:hx+hps])

    @staticmethod
    def _augment(lr, hr, vis):
        if random.random() > 0.5:
            lr  = np.ascontiguousarray(np.fliplr(lr))
            hr  = np.ascontiguousarray(np.fliplr(hr))
            vis = np.ascontiguousarray(np.fliplr(vis))
        if random.random() > 0.5:
            lr  = np.ascontiguousarray(np.flipud(lr))
            hr  = np.ascontiguousarray(np.flipud(hr))
            vis = np.ascontiguousarray(np.flipud(vis))
        k = random.choice([0, 1, 2, 3])
        if k:
            lr  = np.ascontiguousarray(np.rot90(lr, k))
            hr  = np.ascontiguousarray(np.rot90(hr, k))
            vis = np.ascontiguousarray(np.rot90(vis, k))
        return lr, hr, vis

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        lr_p, hr_p, vis_p, img_id = self.triplets[idx]
        lr  = self._load_gray(lr_p)
        hr  = self._load_gray(hr_p)
        vis = self._load_rgb(vis_p)

        exp_h, exp_w = lr.shape[0] * self.scale, lr.shape[1] * self.scale
        if hr.shape[0] != exp_h or hr.shape[1] != exp_w:
            hr = np.array(Image.fromarray((hr * 255).astype(np.uint8)).resize(
                (exp_w, exp_h), Image.BICUBIC), np.float32) / 255.0
        if vis.shape[0] != exp_h or vis.shape[1] != exp_w:
            vis = np.array(Image.fromarray((vis * 255).astype(np.uint8)).resize(
                (exp_w, exp_h), Image.BICUBIC), np.float32) / 255.0

        if self.split == "train":
            lr, hr, vis = self._random_crop(lr, hr, vis)
        if self.aug:
            lr, hr, vis = self._augment(lr, hr, vis)

        lr_t  = torch.from_numpy(lr).unsqueeze(0)
        hr_t  = torch.from_numpy(hr).unsqueeze(0)
        vis_t = torch.from_numpy(vis.transpose(2, 0, 1))
        return lr_t, hr_t, vis_t, img_id


class ThermalGuidedTestDataset(Dataset):
    """
    Test dataset – NO ground truth.
      {root}/thermal/test/guided_x{scale}/LR_x{scale}/  *_th.bmp
      {root}/visible/test/guided_x{scale}/               *_vis.bmp
    """
    def __init__(self, root, scale):
        self.scale = scale
        lr_dir  = os.path.join(root, "thermal", "test", f"guided_x{scale}", f"LR_x{scale}")
        vis_dir = os.path.join(root, "visible", "test", f"guided_x{scale}")

        for d, name in [(lr_dir, "Test LR"), (vis_dir, "Test Vis")]:
            if not os.path.isdir(d):
                print(f"  ⚠ WARNING: {name} directory not found: {d}")

        self.pairs = []
        for lr_path in sorted(glob.glob(os.path.join(lr_dir, "*.*"))):
            th_base  = os.path.splitext(os.path.basename(lr_path))[0]
            vis_base = _thermal_base_to_vis_base(th_base)
            vis_path = _find_file(vis_dir, vis_base)
            if vis_path:
                vis_filename = os.path.basename(vis_path)
                self.pairs.append((lr_path, vis_path, th_base, vis_filename))

        print(f"  [test ] Loaded {len(self.pairs)} pairs  (scale ×{scale})")

    @staticmethod
    def _load_gray(path):
        return np.array(Image.open(path).convert("L"), np.float32) / 255.0

    @staticmethod
    def _load_rgb(path):
        return np.array(Image.open(path).convert("RGB"), np.float32) / 255.0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_p, vis_p, th_base, vis_filename = self.pairs[idx]
        lr  = self._load_gray(lr_p)
        vis = self._load_rgb(vis_p)

        exp_h, exp_w = lr.shape[0] * self.scale, lr.shape[1] * self.scale
        if vis.shape[0] != exp_h or vis.shape[1] != exp_w:
            vis = np.array(Image.fromarray((vis * 255).astype(np.uint8)).resize(
                (exp_w, exp_h), Image.BICUBIC), np.float32) / 255.0

        lr_t  = torch.from_numpy(lr).unsqueeze(0)
        vis_t = torch.from_numpy(vis.transpose(2, 0, 1))
        return lr_t, vis_t, th_base, vis_filename


###############################################################################
#                        8.  SELF-ENSEMBLE (8×)                               #
###############################################################################

@torch.no_grad()
def self_ensemble(model, lr, vis):
    """Average predictions over 8 augmentations (4 rots × 2 flips)."""
    preds = []
    for hflip in (False, True):
        for rot_k in (0, 1, 2, 3):
            a_lr, a_vis = lr.clone(), vis.clone()
            if hflip:
                a_lr  = torch.flip(a_lr, [-1])
                a_vis = torch.flip(a_vis, [-1])
            if rot_k:
                a_lr  = torch.rot90(a_lr,  rot_k, [-2, -1])
                a_vis = torch.rot90(a_vis, rot_k, [-2, -1])

            pred = model(a_lr, a_vis)

            if rot_k:
                pred = torch.rot90(pred, -rot_k, [-2, -1])
            if hflip:
                pred = torch.flip(pred, [-1])
            preds.append(pred)
    return torch.stack(preds).mean(0)


###############################################################################
#                         9.  VALIDATE                                        #
###############################################################################

@torch.no_grad()
def validate(model, loader, device, use_ensemble=False):
    model.eval()
    psnrs, ssims = [], []
    for batch in loader:
        lr, hr, vis = batch[0], batch[1], batch[2]
        lr, hr, vis = lr.to(device), hr.to(device), vis.to(device)
        pred = self_ensemble(model, lr, vis) if use_ensemble else model(lr, vis)
        pred = pred.clamp(0, 1)
        if pred.shape[2:] != hr.shape[2:]:
            pred = F.interpolate(pred, hr.shape[2:], mode="bilinear", align_corners=False)
        for i in range(pred.size(0)):
            p = pred[i, 0].cpu().numpy()
            h = hr[i, 0].cpu().numpy()
            psnrs.append(calc_psnr(h, p, data_range=1.0))
            ssims.append(calc_ssim(h, p, data_range=1.0))
    return float(np.mean(psnrs)), float(np.mean(ssims))


###############################################################################
#                      10.  UPDATED CONFIGS                                   #
###############################################################################

CONFIGS = {
    8: dict(
        nf=64, n_rrdb=12, gc=32, n_rcab_stage=3, n_rcab_final=4,
        bs=4, ps=256, lr=3e-4, epochs=500,
        loss_w=dict(w_char=1.0, w_ssim=0.5, w_perc=0.02, w_edge=0.1)
    ),
    16: dict(
        nf=64, n_rrdb=12, gc=32, n_rcab_stage=3, n_rcab_final=6,
        bs=2, ps=192, lr=3e-4, epochs=600,
        loss_w=dict(w_char=1.0, w_ssim=0.5, w_perc=0.02, w_edge=0.1)
    ),
}


###############################################################################
#                       11.  TRAINING FUNCTION                                #
###############################################################################

def train_model(scale, data_root):
    cfg = CONFIGS[scale]
    out = get_out_dir()
    ckpt_dir = os.path.join(out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"#  TRAINING  ×{scale}   |  {cfg['n_rrdb']} GuidedRRDB  |  {cfg['epochs']} epochs")
    print(f"{'#'*70}")
    print(f"Config: {cfg}")
    print(f"Checkpoints → {ckpt_dir}\n")

    # Data
    train_ds = ThermalGuidedDataset(data_root, scale, "train", cfg["ps"], augment=True)
    val_ds   = ThermalGuidedDataset(data_root, scale, "val",   cfg["ps"], augment=False)
    train_dl = DataLoader(train_ds, cfg["bs"], shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)

    # Model
    model = GuidedThermalSRNet(
        scale, cfg["nf"], cfg["n_rrdb"], cfg["gc"],
        cfg["n_rcab_stage"], cfg["n_rcab_final"]
    ).to(DEVICE)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_total:,} total  |  {n_train:,} trainable  |  "
          f"{n_total*4/1024**2:.1f} MB\n")

    # Loss
    criterion = CombinedLoss(**cfg["loss_w"]).to(DEVICE)

    # Optimizer with differential LR
    guide_ids = set(id(p) for p in model.guide_enc.parameters())
    main_params  = [p for p in model.parameters()
                    if id(p) not in guide_ids and p.requires_grad]
    guide_params = [p for p in model.guide_enc.parameters() if p.requires_grad]

    optimizer = optim.AdamW([
        {"params": main_params,  "lr": cfg["lr"]},
        {"params": guide_params, "lr": cfg["lr"] * 0.1},
    ], weight_decay=1e-4)

    # ★ Scheduler with longer cycles (T_0=100)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-6)

    ema = EMA(model, decay=0.999)

    best_ckpt = os.path.join(ckpt_dir, f"best_x{scale}.pth")
    best_psnr, best_ssim = 0.0, 0.0
    history = {"epoch": [], "loss": [], "psnr": [], "ssim": []}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ep_loss = 0.0

        for batch in train_dl:
            lr, hr, vis = batch[0], batch[1], batch[2]
            lr, hr, vis = lr.to(DEVICE), hr.to(DEVICE), vis.to(DEVICE)

            pred = model(lr, vis)
            if pred.shape[2:] != hr.shape[2:]:
                pred = F.interpolate(pred, hr.shape[2:],
                                     mode="bilinear", align_corners=False)

            loss = criterion(pred, hr)
            if math.isnan(loss.item()):
                print("⚠ NaN loss – skipping batch")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            ema.update(model)
            ep_loss += loss.item()

        scheduler.step()
        avg_loss = ep_loss / max(len(train_dl), 1)

        psnr_val = ssim_val = None
        if epoch % 5 == 0 or epoch <= 3:
            ema.apply(model)
            psnr_val, ssim_val = validate(model, val_dl, DEVICE, use_ensemble=False)
            ema.restore(model)

            improved = ""
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_ssim = ssim_val
                ema.apply(model)
                torch.save(model.state_dict(), best_ckpt)
                ema.restore(model)
                improved = " ★"

            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Ep {epoch:4d}/{cfg['epochs']}  loss={avg_loss:.5f}  "
                  f"PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  "
                  f"best={best_psnr:.2f}/{best_ssim:.4f}  lr={cur_lr:.1e}{improved}")

        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["psnr"].append(psnr_val)
        history["ssim"].append(ssim_val)

    # Final eval with self-ensemble
    print(f"\n  Final evaluation with 8× self-ensemble …")
    model.load_state_dict(torch.load(best_ckpt))
    model.to(DEVICE)
    final_psnr, final_ssim = validate(model, val_dl, DEVICE, use_ensemble=True)
    print(f"  ✓  x{scale}  SE-PSNR = {final_psnr:.2f} dB   SE-SSIM = {final_ssim:.4f}")

    if final_psnr > best_psnr:
        best_psnr = final_psnr
        best_ssim = final_ssim

    return model, history, best_psnr, best_ssim


###############################################################################
#        12.  TEST INFERENCE + SUBMISSION ZIP (ALL test images)               #
###############################################################################

@torch.no_grad()
def run_test_and_create_submission(data_root):
    """
    Run inference on ALL test images for both scales,
    create submission.zip with proper naming.
    """
    out = get_out_dir()

    submission_dir = os.path.join(out, "submission")
    os.makedirs(submission_dir, exist_ok=True)

    all_metrics = {}

    for scale in [8, 16]:
        cfg = CONFIGS[scale]
        ckpt_path = os.path.join(out, "checkpoints", f"best_x{scale}.pth")

        if not os.path.isfile(ckpt_path):
            print(f"\n  ✗ Checkpoint not found: {ckpt_path}")
            continue

        # Load model
        model = GuidedThermalSRNet(
            scale, cfg["nf"], cfg["n_rrdb"], cfg["gc"],
            cfg["n_rcab_stage"], cfg["n_rcab_final"]
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        # Create scale-specific submission folder
        sub_dir = os.path.join(submission_dir, f"x{scale}")
        os.makedirs(sub_dir, exist_ok=True)
        # Clear old files
        for f in glob.glob(os.path.join(sub_dir, "*")):
            os.remove(f)

        # Load test dataset
        test_ds = ThermalGuidedTestDataset(data_root, scale)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

        print(f"\n{'─'*70}")
        print(f"  ×{scale} TEST SET — {len(test_ds)} images")
        print(f"{'─'*70}")
        print(f"  {'#':>4s}  {'Thermal ID':<25s}  →  {'Output Name':<25s}")
        print(f"  {'─'*4}  {'─'*25}     {'─'*25}")

        for idx, (lr, vis, th_base, vis_filename) in enumerate(test_dl):
            lr, vis = lr.to(DEVICE), vis.to(DEVICE)

            # 8× self-ensemble
            pred = self_ensemble(model, lr, vis).clamp(0, 1)

            sr_np = pred[0, 0].cpu().numpy()
            sr_uint8 = (np.clip(sr_np, 0, 1) * 255).astype(np.uint8)

            # Output filename: use visible filename, ensure .bmp extension
            out_name = os.path.splitext(vis_filename[0])[0] + ".bmp"
            out_path = os.path.join(sub_dir, out_name)

            Image.fromarray(sr_uint8, mode="L").save(out_path, format="BMP")

            print(f"  {idx+1:4d}  {th_base[0]:<25s}  →  {out_name:<25s}")

        print(f"\n  ✓ Saved {len(test_ds)} SR images → {sub_dir}/")

        # ── Also compute validation metrics ──
        val_ds = ThermalGuidedDataset(data_root, scale, "val", cfg["ps"], augment=False)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

        print(f"\n  Computing validation metrics for ×{scale}...")
        psnr_sr, ssim_sr = validate(model, val_dl, DEVICE, use_ensemble=True)
        all_metrics[scale] = {"psnr": psnr_sr, "ssim": ssim_sr}
        print(f"  Val PSNR={psnr_sr:.2f} dB   SSIM={ssim_sr:.4f}")

    # ── Create submission.zip ──
    x8_dir  = os.path.join(submission_dir, "x8")
    x16_dir = os.path.join(submission_dir, "x16")
    zip_path = os.path.join(out, "submission.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder_name, folder_path in [("x8", x8_dir), ("x16", x16_dir)]:
            if os.path.isdir(folder_path):
                for fname in sorted(os.listdir(folder_path)):
                    full_path = os.path.join(folder_path, fname)
                    arcname   = os.path.join(folder_name, fname)
                    zf.write(full_path, arcname)

    # Verify
    with zipfile.ZipFile(zip_path, "r") as zf:
        contents = sorted(zf.namelist())

    n_x8  = [c for c in contents if c.startswith("x8/")]
    n_x16 = [c for c in contents if c.startswith("x16/")]

    print(f"\n{'='*70}")
    print(f"  ✅  SUBMISSION ZIP CREATED")
    print(f"{'='*70}")
    print(f"  Path  : {zip_path}")
    print(f"  Size  : {os.path.getsize(zip_path) / 1024:.1f} KB")
    print(f"  x8/   : {len(n_x8)} files")
    print(f"  x16/  : {len(n_x16)} files")
    print(f"  Total : {len(contents)} files")

    # ── Print validation summary ──
    print(f"\n{'='*70}")
    print(f"  VALIDATION METRICS (8× self-ensemble)")
    print(f"{'='*70}")
    for scale in [8, 16]:
        if scale in all_metrics:
            m = all_metrics[scale]
            print(f"  ×{scale}:  PSNR = {m['psnr']:.2f} dB   SSIM = {m['ssim']:.4f}")

    # ── List zip contents ──
    print(f"\n  Zip contents:")
    for c in contents[:10]:
        print(f"    {c}")
    if len(contents) > 10:
        print(f"    ... and {len(contents)-10} more files")

    print(f"\n  📦 Ready to submit: {zip_path}")

    return zip_path, all_metrics


###############################################################################
#                       13.  DISPLAY SUMMARY                                  #
###############################################################################

def display_results(h8, h16, p8, s8, p16, s16):
    out = get_out_dir()

    print(f"\n{'='*70}")
    print("                     FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  ×8   PSNR = {p8:.2f} dB    SSIM = {s8:.4f}")
    print(f"  ×16  PSNR = {p16:.2f} dB    SSIM = {s16:.4f}")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for hist, label in [(h8, "×8"), (h16, "×16")]:
        axes[0].plot(hist["epoch"], hist["loss"], label=label, alpha=0.7)
        eps = [e for e, v in zip(hist["epoch"], hist["psnr"]) if v is not None]
        pvs = [v for v in hist["psnr"] if v is not None]
        svs = [v for v in hist["ssim"] if v is not None]
        axes[1].plot(eps, pvs, "o-", label=label, markersize=2)
        axes[2].plot(eps, svs, "o-", label=label, markersize=2)

    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    axes[0].legend(); axes[0].grid(True, alpha=.3)
    axes[1].set(xlabel="Epoch", ylabel="PSNR (dB)", title="Val PSNR")
    axes[1].legend(); axes[1].grid(True, alpha=.3)
    axes[2].set(xlabel="Epoch", ylabel="SSIM", title="Val SSIM")
    axes[2].legend(); axes[2].grid(True, alpha=.3)
    plt.tight_layout()

    curves_path = os.path.join(out, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"  Saved {curves_path}")

    summary_path = os.path.join(out, "final_summary.txt")
    with open(summary_path, "w") as f:
        f.write("FINAL RESULTS SUMMARY (v2 - GuidedRRDB)\n")
        f.write(f"{'='*50}\n")
        f.write(f"×8   PSNR = {p8:.2f} dB    SSIM = {s8:.4f}\n")
        f.write(f"×16  PSNR = {p16:.2f} dB    SSIM = {s16:.4f}\n")
        f.write(f"{'='*50}\n\n")
        f.write("Key changes in v2:\n")
        f.write("  - GuidedRRDB: guide features injected into every trunk block\n")
        f.write("  - Larger patches (256 for x8, 192 for x16)\n")
        f.write("  - Deeper trunk (12 GuidedRRDB blocks)\n")
        f.write("  - Stronger SSIM loss weight (0.5)\n")
        f.write("  - Stronger edge loss weight (0.1)\n")
        f.write("  - Longer LR schedule cycles (T_0=100)\n")
    print(f"  Saved {summary_path}")


###############################################################################
#                              14.  MAIN                                      #
###############################################################################

if __name__ == "__main__":
    DATA_ROOT = "/home/rsnfh/Downloads/CIDIS Dataset"

    if not os.path.isdir(DATA_ROOT):
        print(f"ERROR: dataset not found at {DATA_ROOT}")
        sys.exit(1)

    # ──── Set output directory (v2 folder) ────
    set_out_dir(DATA_ROOT)

    # ──────────────── Verify directory structure ────────────────
    print("\n" + "="*70)
    print("  VERIFYING DATASET DIRECTORY STRUCTURE")
    print("="*70)
    expected = {
        "thermal/train/GT":                    "HR thermal GT (train)",
        "thermal/train/LR_x8":                 "LR thermal ×8 (train)",
        "thermal/train/LR_x16":                "LR thermal ×16 (train)",
        "thermal/val/GT":                      "HR thermal GT (val)",
        "thermal/val/LR_x8":                   "LR thermal ×8 (val)",
        "thermal/val/LR_x16":                  "LR thermal ×16 (val)",
        "visible/train":                       "HR visible (train)",
        "visible/val":                         "HR visible (val)",
        "thermal/test/guided_x8/LR_x8":        "LR thermal ×8 (test)",
        "thermal/test/guided_x16/LR_x16":      "LR thermal ×16 (test)",
        "visible/test/guided_x8":              "HR visible ×8 (test)",
        "visible/test/guided_x16":             "HR visible ×16 (test)",
    }
    all_ok = True
    for subdir, desc in expected.items():
        full = os.path.join(DATA_ROOT, subdir)
        exists = os.path.isdir(full)
        n_files = len(glob.glob(os.path.join(full, "*.*"))) if exists else 0
        status = f"✓ {n_files:4d} files" if exists else "✗ MISSING"
        print(f"  {status}  {subdir:<45s}  ({desc})")
        if not exists:
            all_ok = False

    # ──── File pairing check ────
    print("\n  FILE-NAME PAIRING CHECK (first 3 from train ×8):")
    s_lr  = os.path.join(DATA_ROOT, "thermal", "train", "LR_x8")
    s_gt  = os.path.join(DATA_ROOT, "thermal", "train", "GT")
    s_vis = os.path.join(DATA_ROOT, "visible", "train")
    for lr_p in sorted(glob.glob(os.path.join(s_lr, "*.*")))[:3]:
        th_base  = os.path.splitext(os.path.basename(lr_p))[0]
        vis_base = _thermal_base_to_vis_base(th_base)
        gt_ok  = _find_file(s_gt, th_base) is not None
        vis_ok = _find_file(s_vis, vis_base) is not None
        print(f"    LR: {th_base}  →  GT: {'✓' if gt_ok else '✗'}  "
              f" Vis({vis_base}): {'✓' if vis_ok else '✗'}")

    # ──────────────── Print config summary ────────────────
    print("\n" + "="*70)
    print("  CONFIGURATION SUMMARY (v2)")
    print("="*70)
    for scale in [8, 16]:
        cfg = CONFIGS[scale]
        print(f"\n  ×{scale}:")
        print(f"    Trunk:     {cfg['n_rrdb']} GuidedRRDB blocks")
        print(f"    Patch:     {cfg['ps']}×{cfg['ps']} (HR)")
        print(f"    Batch:     {cfg['bs']}")
        print(f"    LR:        {cfg['lr']}")
        print(f"    Epochs:    {cfg['epochs']}")
        print(f"    Losses:    char={cfg['loss_w']['w_char']}, "
              f"ssim={cfg['loss_w']['w_ssim']}, "
              f"perc={cfg['loss_w']['w_perc']}, "
              f"edge={cfg['loss_w']['w_edge']}")

    # ──────────────── Train ×8 ────────────────
    model_x8, hist_x8, psnr_x8, ssim_x8 = train_model(scale=8, data_root=DATA_ROOT)

    # ──────────────── Train ×16 ────────────────
    model_x16, hist_x16, psnr_x16, ssim_x16 = train_model(scale=16, data_root=DATA_ROOT)

    # ──────────────── Training curves + summary ────────────────
    display_results(hist_x8, hist_x16, psnr_x8, ssim_x8, psnr_x16, ssim_x16)

    # ──────────────── Test inference + submission zip ────────────────
    print("\n" + "#"*70)
    print("#  TEST INFERENCE + SUBMISSION ZIP")
    print("#"*70)

    zip_path, metrics = run_test_and_create_submission(DATA_ROOT)

    # ──────────────── Final message ────────────────
    print(f"\n{'='*70}")
    print(f"  ✅  ALL DONE (v2 - GuidedRRDB)")
    print(f"{'='*70}")
    print(f"  All outputs   → {get_out_dir()}")
    print(f"  Submission    → {zip_path}")
    print(f"{'='*70}")
