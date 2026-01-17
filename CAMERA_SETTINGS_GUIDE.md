# CAMERA_SETTINGS_GUIDE.md

## Purpose
This document describes the camera configuration philosophy, shared defaults, and concrete usage patterns used throughout the **Rasp** project.

The project prioritizes stable data flow and predictable detection behavior over image aesthetics. Camera tuning is intentionally conservative and global where possible, with targeted overrides only when required by platform or detection domain.

---

## Core Philosophy
Rasp is an edge AI detection system, not a video production pipeline.

Design priorities:
- Deterministic frame timing
- Stable luminance over time
- Bounded CPU usage
- Predictable motion metrics

Most analysis operates on grayscale or luminance-derived frames. As a result:
- Color fidelity is secondary
- Exposure stability is critical
- Motion blur and noise directly affect detection reliability

Camera tuning is deferred until communications, storage, inference pacing, and motion gating are verified.

---

## Project-Wide Defaults

### Resolution
- Static cameras: **1280×720**
- Drone / mobile: **960×540** (or lower)

Higher resolutions increase latency and CPU load and are avoided unless required.

### Frame Rate (FPS)
- Static cameras: **15–20 FPS**
- Drone / mobile: **25–30 FPS at reduced resolution**

Higher FPS on mobile platforms reduces motion blur and improves temporal alignment, not visual smoothness.

### Exposure
- Auto Exposure (AE) enabled by default
- Fixed exposure only if instability is observed

Exposure jumps resemble motion and break motion gating.

### Gain / ISO
- Moderate gain allowed in low light
- Avoid unbounded gain ranges
- Noise is preferable to blur

### Auto White Balance (AWB)
- Enabled by default
- Disabled only in controlled lighting

AWB affects luminance distribution even in grayscale pipelines.

### Sharpness / Contrast
- Neutral or minimal
- Tune only after detection is stable

---

## Concrete CLI Examples

### Humans
```bash
python3 9_AICAM_Humans.py --width 1280 --height 720 --fps 20 --confidence 0.50 --confirm-frames 3 --cooldown 15
```

### Faces
```bash
python3 _10.1_AICAM_Faces.py --width 1280 --height 720 --fps 20 --confidence 0.55 --confirm-frames 3 --cooldown 15
```

### Animals
```bash
python3 _11_AICAM_Animals.py --width 1280 --height 720 --fps 20 --confidence 0.50 --confirm-frames 3 --cooldown 20
```

### Birds
```bash
python3 _12_AICAM_Birds.py --width 1280 --height 720 --fps 20 --confidence 0.60 --confirm-frames 4 --cooldown 15
```

### Drone
```bash
python3 AICAM_Drone_picamera2.py --width 960 --height 540 --fps 30 --max-infer-fps 8 --motion-roi-scale 0.5 --adaptive-skip
```

---

## Field Checklist
- Verify FPS stability
- Watch for exposure jumps
- Prefer noise over blur
- Tune detection thresholds only after camera stability
