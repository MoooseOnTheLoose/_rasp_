# PICAMERA2_RPICAM_TUNING.md

## Purpose
This document provides practical guidance for tuning cameras in Rasp using **rpicam** tools and **Picamera2**.

---

## rpicam Validation

```bash
rpicam-hello
rpicam-vid --width 1280 --height 720 --framerate 20 --timeout 0
```

Observe exposure behavior, motion blur, and frame pacing.

### Shutter Testing
```bash
rpicam-vid --shutter 3000
rpicam-vid --shutter 6000
rpicam-vid --shutter 12000
```

---

## Picamera2 Translation

### Baseline Controls
```python
picam2.set_controls({"AeEnable": True, "AwbEnable": True})
```

### Fixed Exposure (Only If Needed)
```python
picam2.set_controls({"AeEnable": False, "ExposureTime": 8000, "AnalogueGain": 2.0})
```

### Frame Timing
```python
picam2.set_controls({"FrameDurationLimits": (50000, 50000)})
```

---

## Recommended Profiles
- Day static: 1280×720 @ 20 FPS
- Night static: 1280×720 @ 15 FPS
- Drone: 960×540 @ 30 FPS

---

## Workflow
1. Validate with rpicam
2. Set global defaults
3. Run detection
4. Adjust camera only if unstable
5. Tune detection last
