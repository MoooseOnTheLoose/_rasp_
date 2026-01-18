#!/usr/bin/env python3
"""AICAM Drone (YOLO-only, ONNX Runtime).

Single-purpose drone clip recorder aligned with the _Rasp_ AICAM scripts.

Design
- Picamera2 capture -> YOLO ONNX inference (onnxruntime CPU) -> confirm -> trigger -> record -> cooldown
- One model path, one pipeline. No Caffe-SSD or OpenCV-DNN ONNX import.

Artifacts (base_dir)
- logs/aicam_drone.log        : rotating operational log
- logs/events.jsonl           : rotating structured events
- logs/events.log             : human-friendly text events
- clips/<clip>.mp4            : recorded clip
- clips/<clip>.json           : per-clip metadata

Storage location default: build_arg_parser(): line 477, 479 <--- Change

Annotation
- --annotate-preview                    draw boxes/conf on preview only
- --annotate-clips {none,secondary,burnin}
  - none      : clean clip only
  - secondary : clean clip + *_annot.mp4
  - burnin    : overlays on primary clip

Model contract (fail-fast)
- Input  : float32 [1,3,640,640]
- Output : float32 [1,5,8400]
  Interpreted as 8400 candidates: [cx, cy, w, h, score] in model input coords.

Run example:
python3 AICAM_Drones_YoloOnnx.py \
        --yolo-model /home/usr/models/best.onnx \
        --annotate-clips secondary \
        --conf 0.35
        --min-area-px 2000 \
        --confirm-hits 4 \
        --confirm-window-s 1.5
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import logging.handlers
import os
import shutil
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None

try:
    from picamera2 import Picamera2  # type: ignore
except Exception:  # pragma: no cover
    Picamera2 = None

Box = Tuple[int, int, int, int]  # x1,y1,x2,y2


# ----------------------------
# Time helpers
# ----------------------------

def ts_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ts_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ----------------------------
# Storage selection
# ----------------------------

def _is_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".aicam_write_test"
        with open(test, "w", encoding="utf-8") as f:
            f.write("ok")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def pick_base_dir(mount: str, base_subdir: str, fallback_dir: str) -> Path:
    """Prefer mount if present + writable, else fallback."""
    candidates: List[Path] = []
    if mount:
        mp = Path(mount)
        if mp.exists() and mp.is_dir():
            candidates.append(mp / base_subdir)
    candidates.append(Path(fallback_dir) / base_subdir)

    base: Optional[Path] = None
    for c in candidates:
        if _is_writable_dir(c):
            base = c
            break

    if base is None:
        # Last resort: current directory
        base = Path.cwd() / base_subdir
        base.mkdir(parents=True, exist_ok=True)

    (base / "clips").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "tmp").mkdir(parents=True, exist_ok=True)
    return base


def free_space_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(str(path))
        return float(usage.free) / (1024.0 ** 3)
    except Exception:
        return 0.0


# ----------------------------
# Logging
# ----------------------------

def setup_rotating_logger(name: str, path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if logger.handlers:
        return logger

    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class JsonlEvents:
    def __init__(self, path: Path, max_bytes: int = 2_000_000, backup_count: int = 5) -> None:
        self._logger = logging.getLogger("events_aicam_drone_jsonl")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(fh)

    def emit(self, payload: Dict) -> None:
        try:
            self._logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            pass


def events_text_line(event: str, **kv: object) -> str:
    parts = [event]
    for k in sorted(kv.keys()):
        v = kv[k]
        parts.append(f"{k}={v}")
    return " ".join(parts)


# ----------------------------
# Names
# ----------------------------

def load_names(names_path: Optional[str]) -> List[str]:
    if not names_path:
        return ["drone"]
    p = Path(names_path)
    if not p.exists():
        return ["drone"]
    out: List[str] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out or ["drone"]


# ----------------------------
# Letterbox + mapping
# ----------------------------

def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)):
    """Resize with padding to meet stride-multiple constraints.

    Returns (im_out, ratio, (pad_w, pad_h)) where:
    - ratio is a scalar gain applied to original coords
    - pad is padding added (left/right, top/bottom split equally in this implementation)
    """
    h0, w0 = im.shape[:2]
    new_w, new_h = new_shape
    gain = min(new_w / w0, new_h / h0)
    resize_w = int(round(w0 * gain))
    resize_h = int(round(h0 * gain))
    im_resized = cv2.resize(im, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resize_w
    pad_h = new_h - resize_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    im_out = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_out, gain, (left, top)


def clip_box_to_frame(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def unletterbox_xyxy(xyxy: np.ndarray, gain: float, pad: Tuple[int, int], orig_wh: Tuple[int, int]) -> np.ndarray:
    """Map xyxy in 640-letterbox coords back to original image coords."""
    pad_x, pad_y = pad
    out = xyxy.copy().astype(np.float32)
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out /= gain
    w0, h0 = orig_wh
    out[:, 0] = np.clip(out[:, 0], 0, w0 - 1)
    out[:, 2] = np.clip(out[:, 2], 0, w0 - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h0 - 1)
    out[:, 3] = np.clip(out[:, 3], 0, h0 - 1)
    return out


# ----------------------------
# NMS
# ----------------------------

def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Pure numpy NMS. boxes: Nx4 float, scores: N."""
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


# ----------------------------
# Detection
# ----------------------------

@dataclass
class Detection:
    cls_name: str
    conf: float
    box: Box


class YoloOnnxDrone:
    """YOLO ONNX (best.onnx) detector using ONNX Runtime.

    Fail-fast on model I/O layout to prevent silent mis-decoding.
    """

    def __init__(
        self,
        model_path: str,
        class_names: Sequence[str],
        conf_thres: float,
        iou_thres: float,
        ort_intra_threads: int,
        ort_inter_threads: int,
        ort_log_severity: int,
        log: logging.Logger,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed for this python. Install via apt (python3-onnxruntime) or pip, then rerun."
            )
        self.model_path = str(model_path)
        self.class_names = list(class_names) if class_names else ["drone"]
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.log = log

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(max(1, ort_intra_threads))
        so.inter_op_num_threads = int(max(1, ort_inter_threads))
        so.log_severity_level = int(ort_log_severity)

        self.sess = ort.InferenceSession(self.model_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()
        self.out = self.sess.get_outputs()

        if len(self.inp) != 1:
            raise RuntimeError(f"Expected 1 input, got {len(self.inp)}: {[i.name for i in self.inp]}")
        if len(self.out) != 1:
            raise RuntimeError(f"Expected 1 output, got {len(self.out)}: {[o.name for o in self.out]}")

        in_shape = list(self.inp[0].shape)
        out_shape = list(self.out[0].shape)

        # Strict contract for this project.
        if in_shape != [1, 3, 640, 640]:
            raise RuntimeError(f"Unsupported input shape {in_shape}; expected [1,3,640,640]")
        if out_shape != [1, 5, 8400]:
            raise RuntimeError(f"Unsupported output shape {out_shape}; expected [1,5,8400]")

        self.in_name = self.inp[0].name
        self.out_name = self.out[0].name

        self.log.info(f"ORT model loaded | in={self.in_name} {in_shape} | out={self.out_name} {out_shape}")

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        # letterbox to 640
        lb, gain, pad = letterbox(frame_bgr, (640, 640))
        # BGR->RGB, float32 0-1, NCHW
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x, gain, pad

    def infer(self, frame_bgr: np.ndarray, min_area_px: int = 0) -> List[Detection]:
        h0, w0 = frame_bgr.shape[:2]
        x, gain, pad = self._preprocess(frame_bgr)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]

        # y is [1,5,8400] -> (8400,5)
        if y.ndim != 3 or list(y.shape) != [1, 5, 8400]:
            raise RuntimeError(f"Unexpected output at runtime: shape={getattr(y, 'shape', None)}")
        y = np.transpose(y[0], (1, 0))

        cx = y[:, 0]
        cy = y[:, 1]
        w = y[:, 2]
        h = y[:, 3]
        scores = y[:, 4]

        keep = scores >= self.conf_thres
        if not np.any(keep):
            return []

        cx = cx[keep]
        cy = cy[keep]
        w = w[keep]
        h = h[keep]
        scores = scores[keep]

        # xyxy in letterbox space
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # map back to original
        boxes = unletterbox_xyxy(boxes, gain, pad, (w0, h0))

        # NMS
        idxs = nms_xyxy(boxes, scores, self.iou_thres)

        out: List[Detection] = []
        for i in idxs:
            bx = boxes[i]
            b = (int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]))
            b = clip_box_to_frame(b, w0, h0)
            if min_area_px > 0:
                area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                if area < min_area_px:
                    continue
            out.append(Detection(cls_name=self.class_names[0], conf=float(scores[i]), box=b))
        return out


# ----------------------------
# Annotation
# ----------------------------

def draw_overlay(frame: np.ndarray, det: Optional[Detection], state: str, fps: Optional[float] = None) -> np.ndarray:
    if det is None:
        # still show state
        text = state if fps is None else f"{state} | {fps:.1f} fps"
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    x1, y1, x2, y2 = det.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{det.cls_name} {det.conf:.2f}"
    cv2.putText(frame, label, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    text = state if fps is None else f"{state} | {fps:.1f} fps"
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ----------------------------
# Recording
# ----------------------------

def atomic_write_json(path: Path, payload: Dict, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f".{path.name}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp_path), str(path))


def open_writer(path: Path, fps: float, size_wh: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size_wh)


# ----------------------------
# Main loop
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AICAM drone recorder (YOLO-only ONNX Runtime)")

    # Storage
    p.add_argument("--mount", default="/media/user/deskView", help="Preferred mount root") # Change default mount location.
    p.add_argument("--base-subdir", default="aicam_drone", help="Subdir under mount/fallback")
    p.add_argument("--fallback-dir", default="/home/user/aicam_drone", help="Fallback base directory") # Change default mount location
    p.add_argument("--min-free-gb", type=float, default=10.0, help="Skip trigger if free space below this")

    # Identity/logging
    p.add_argument("--device-id", default="aircam_01", help="Device identifier for events")
    p.add_argument("--log-level", default="INFO", help="INFO/DEBUG")

    # Camera
    p.add_argument("--fps", type=float, default=30.0, help="Recording fps")
    p.add_argument("--width", type=int, default=1920, help="Capture width")
    p.add_argument("--height", type=int, default=1080, help="Capture height")
    p.add_argument("--preview", action="store_true", help="Show live preview window")

    # Detector/model
    p.add_argument("--yolo-model", required=True, help="Path to best.onnx")
    p.add_argument("--yolo-names", default=None, help="Path to drone.names (one label per line)")
    p.add_argument("--classes", default="drone", help="Comma-separated allowed class names (default drone)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--min-area-px", type=int, default=0, help="Minimum bbox area in pixels")

    # Trigger/clip
    p.add_argument("--clip-len-s", type=float, default=10.0, help="Recorded clip length")
    p.add_argument("--preroll-s", type=float, default=2.0, help="Seconds of preroll buffer")
    p.add_argument("--cooldown-s", type=float, default=10.0, help="Seconds to wait after trigger")
    p.add_argument("--confirm-hits", type=int, default=4, help="Detections required in window")
    p.add_argument("--confirm-window-s", type=float, default=1.5, help="Window in seconds")
    p.add_argument("--max-infer-fps", type=float, default=12.0, help="Cap inference rate")

    # ORT
    p.add_argument("--ort-intra-threads", type=int, default=2, help="ORT intra-op threads")
    p.add_argument("--ort-inter-threads", type=int, default=1, help="ORT inter-op threads")
    p.add_argument("--ort-log-severity", type=int, default=3, help="0=verbose..3=error")

    # Annotation
    p.add_argument("--annotate-preview", action="store_true", help="Draw overlays on preview")
    p.add_argument(
        "--annotate-clips",
        choices=["none", "secondary", "burnin"],
        default="none",
        help="Clip annotation mode",
    )

    # Utilities
    p.add_argument("--selftest", action="store_true", help="Load model and exit")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    base = pick_base_dir(args.mount, args.base_subdir, args.fallback_dir)
    logs_dir = base / "logs"
    clips_dir = base / "clips"
    tmp_dir = base / "tmp"

    op_log = setup_rotating_logger("aicam_drone", logs_dir / "aicam_drone.log", level=args.log_level)
    events = JsonlEvents(logs_dir / "events.jsonl")
    events_text = setup_rotating_logger("aicam_drone_events_text", logs_dir / "events.log", level="INFO")

    session = ts_utc_compact()
    allowed = {s.strip() for s in str(args.classes).split(",") if s.strip()}

    op_log.info(f"start | session={session} device_id={args.device_id} base={base}")
    events_text.info(events_text_line("START", session=session, device_id=args.device_id, base=str(base)))
    events.emit({"ts": ts_utc_iso(), "event": "START", "session": session, "device_id": args.device_id, "base": str(base)})

    class_names = load_names(args.yolo_names)

    detector = YoloOnnxDrone(
        model_path=args.yolo_model,
        class_names=class_names,
        conf_thres=args.conf,
        iou_thres=args.iou,
        ort_intra_threads=args.ort_intra_threads,
        ort_inter_threads=args.ort_inter_threads,
        ort_log_severity=args.ort_log_severity,
        log=op_log,
    )

    if args.selftest:
        print("OK: model loaded")
        return 0

    if Picamera2 is None:
        raise RuntimeError(
            "picamera2 is not available in this python. Install python3-picamera2 via apt and run outside a venv."
        )

    # Camera setup
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(main={"size": (args.width, args.height), "format": "RGB888"})
    picam2.configure(cfg)
    picam2.start()

    w0, h0 = args.width, args.height
    fps = float(args.fps)

    # Preroll buffer holds raw BGR frames (for saving)
    preroll: Deque[np.ndarray] = collections.deque(maxlen=max(1, int(fps * float(args.preroll_s))))

    # Confirm hits store timestamps
    hits: Deque[float] = collections.deque()

    last_infer = 0.0
    infer_period = 1.0 / max(0.1, float(args.max_infer_fps))

    state = "STANDBY"
    overlay_det: Optional[Detection] = None

    # Recording state
    recording = False
    rec_start = 0.0
    clip_path: Optional[Path] = None
    clip_meta_path: Optional[Path] = None
    writer: Optional[cv2.VideoWriter] = None
    writer_annot: Optional[cv2.VideoWriter] = None
    annot_path: Optional[Path] = None
    preroll_det_snapshot: Optional[Detection] = None

    cooldown_until = 0.0

    # graceful shutdown
    stop_flag = {"stop": False}

    def _handle_sig(_sig, _frm):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    def _finalize_clip(reason: str) -> None:
        nonlocal recording, writer, writer_annot, clip_path, clip_meta_path, annot_path
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if writer_annot is not None:
            try:
                writer_annot.release()
            except Exception:
                pass

        writer = None
        writer_annot = None
        recording = False

        if clip_path and clip_meta_path:
            payload = {
                "ts": ts_utc_iso(),
                "session": session,
                "device_id": args.device_id,
                "model": os.path.basename(args.yolo_model),
                "clip": str(clip_path),
                "annot_clip": str(annot_path) if annot_path else None,
                "reason": reason,
            }
            atomic_write_json(clip_meta_path, payload, tmp_dir)

        events_text.info(
            events_text_line(
                "CLIP_END",
                session=session,
                device_id=args.device_id,
                clip=str(clip_path) if clip_path else "",
                annot_clip=str(annot_path) if annot_path else "",
                reason=reason,
            )
        )
        events.emit(
            {
                "ts": ts_utc_iso(),
                "event": "CLIP_END",
                "session": session,
                "device_id": args.device_id,
                "clip": str(clip_path) if clip_path else None,
                "annot_clip": str(annot_path) if annot_path else None,
                "reason": reason,
            }
        )

        clip_path = None
        clip_meta_path = None
        annot_path = None

    try:
        while not stop_flag["stop"]:
            # capture
            frame_rgb = picam2.capture_array()  # RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            preroll.append(frame_bgr)

            now = time.monotonic()

            # cooldown
            if now < cooldown_until:
                state = "COOLDOWN"
                overlay_det = None
            else:
                # inference throttled
                dets: List[Detection] = []
                if (now - last_infer) >= infer_period:
                    last_infer = now
                    dets = detector.infer(frame_bgr, min_area_px=int(args.min_area_px))

                # keep best det in allowed set
                best: Optional[Detection] = None
                for d in dets:
                    if d.cls_name in allowed:
                        if best is None or d.conf > best.conf:
                            best = d

                overlay_det = best

                # Update hit window
                if best is not None:
                    hits.append(now)
                # prune
                while hits and (now - hits[0]) > float(args.confirm_window_s):
                    hits.popleft()

                # state machine (minimal)
                if recording:
                    state = "RECORD"
                else:
                    state = "STANDBY" if not hits else "ACQUIRE"

                # Trigger
                if (not recording) and len(hits) >= int(args.confirm_hits):
                    # disk guard
                    fs = free_space_gb(base)
                    if fs < float(args.min_free_gb):
                        events_text.info(
                            events_text_line(
                                "SKIP",
                                session=session,
                                device_id=args.device_id,
                                reason="low_disk",
                                free_gb=f"{fs:.2f}",
                            )
                        )
                        events.emit(
                            {
                                "ts": ts_utc_iso(),
                                "event": "SKIP",
                                "session": session,
                                "device_id": args.device_id,
                                "reason": "low_disk",
                                "free_gb": fs,
                            }
                        )
                        hits.clear()
                    else:
                        # start clip
                        ts = ts_utc_compact()
                        clip_name = f"drone_{ts}.mp4"
                        clip_path = clips_dir / clip_name
                        clip_meta_path = clips_dir / f"drone_{ts}.json"

                        # Decide writers
                        write_clean = True
                        write_annot_primary = (args.annotate_clips == "burnin")
                        write_secondary_annot = (args.annotate_clips == "secondary")

                        annot_path = None
                        if write_secondary_annot:
                            annot_path = clips_dir / f"drone_{ts}_annot.mp4"

                        # Open writers
                        writer = open_writer(clip_path, fps=fps, size_wh=(w0, h0))
                        if write_secondary_annot:
                            writer_annot = open_writer(annot_path, fps=fps, size_wh=(w0, h0))  # type: ignore[arg-type]
                        else:
                            writer_annot = None

                        recording = True
                        rec_start = now
                        preroll_det_snapshot = best

                        # Write preroll
                        for fr in list(preroll):
                            if write_clean and writer is not None:
                                if write_annot_primary and preroll_det_snapshot is not None:
                                    fr_out = draw_overlay(fr.copy(), preroll_det_snapshot, state="RECORD")
                                    writer.write(fr_out)
                                else:
                                    writer.write(fr)

                            if write_secondary_annot and writer_annot is not None:
                                fr_a = draw_overlay(fr.copy(), preroll_det_snapshot, state="RECORD")
                                writer_annot.write(fr_a)

                        events_text.info(
                            events_text_line(
                                "TRIGGER",
                                session=session,
                                device_id=args.device_id,
                                clip=str(clip_path),
                                annot_clip=str(annot_path) if annot_path else "",
                                conf=f"{best.conf:.3f}" if best else "",
                                box=str(best.box) if best else "",
                                hits=len(hits),
                            )
                        )
                        events.emit(
                            {
                                "ts": ts_utc_iso(),
                                "event": "TRIGGER",
                                "session": session,
                                "device_id": args.device_id,
                                "clip": str(clip_path),
                                "annot_clip": str(annot_path) if annot_path else None,
                                "conf": float(best.conf) if best else None,
                                "box": list(best.box) if best else None,
                                "hits": len(hits),
                            }
                        )
                        hits.clear()

            # If recording, write frames
            if recording and writer is not None and clip_path is not None:
                elapsed = time.monotonic() - rec_start

                write_annot_primary = (args.annotate_clips == "burnin")
                write_secondary_annot = (args.annotate_clips == "secondary")

                if write_annot_primary and overlay_det is not None:
                    writer.write(draw_overlay(frame_bgr.copy(), overlay_det, state="RECORD"))
                else:
                    writer.write(frame_bgr)

                if write_secondary_annot and writer_annot is not None:
                    writer_annot.write(draw_overlay(frame_bgr.copy(), overlay_det, state="RECORD"))

                if elapsed >= float(args.clip_len_s):
                    _finalize_clip("clip_len")
                    cooldown_until = time.monotonic() + float(args.cooldown_s)
                    preroll_det_snapshot = None

            # Preview
            if args.preview:
                disp = frame_bgr.copy()
                if args.annotate_preview:
                    disp = draw_overlay(disp, overlay_det, state=state)
                cv2.imshow("AICAM Drone", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        # graceful exit
        if recording:
            _finalize_clip("shutdown")

    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        events_text.info(events_text_line("STOP", session=session, device_id=args.device_id))
        events.emit({"ts": ts_utc_iso(), "event": "STOP", "session": session, "device_id": args.device_id})
        op_log.info(f"stop | session={session}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())