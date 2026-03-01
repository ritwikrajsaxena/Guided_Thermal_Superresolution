# Guided Thermal Image Super-Resolution

## Task
Super-resolve low-resolution (LR) thermal images at **×8** and **×16** scale factors,
guided by aligned high-resolution (HR) visible images.

| Scale | Input (LR Thermal) | Guide (HR Visible) | Output (HR Thermal) |
|-------|--------------------|--------------------|---------------------|
| ×8    | ~60×80 px          | ~480×640 px        | ~480×640 px         |
| ×16   | ~30×40 px          | ~480×640 px        | ~480×640 px         |

---

## Method Overview

### Architecture: Guided Thermal SR Network (GuidedTSRNet)
