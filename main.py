"""
CP Plus CCTV — multi-channel CV pipeline.

Usage:
    python main.py                    # display + face recognition + alerts
    python main.py --no-display       # headless
    python main.py --no-recognize     # skip face recognition, alert on all persons
    python main.py --channels 1 2     # specific channels
"""

import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # suppress TF/absl noise

import argparse
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
from detector import Detector
from stream import CameraStream

def _setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    fh = logging.FileHandler(config.LOG_FILE)
    fh.setFormatter(fmt)
    root.addHandler(fh)


_setup_logging()
log = logging.getLogger(__name__)


def build_grid(frames: dict[int, np.ndarray], cell_w: int, cell_h: int) -> np.ndarray:
    channels = sorted(frames)
    cells = []
    for ch in channels:
        cell = cv2.resize(frames[ch], (cell_w, cell_h))
        cv2.putText(cell, f"CH{ch}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cells.append(cell)
    while len(cells) % 2 != 0:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
    rows = [np.hstack(cells[i:i+2]) for i in range(0, len(cells), 2)]
    return np.vstack(rows)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_snapshot(channel: int, frame: np.ndarray, label: str,
                  last_alert: dict, cooldown: int, snap_dir: Path) -> bool:
    now = time.time()
    if now - last_alert.get(channel, 0) < cooldown:
        return False
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"ch{channel}_{_ts()}.jpg"
    cv2.imwrite(str(path), frame)
    log.info("[ch%d] ALERT (%s) → %s", channel, label, path.name)
    last_alert[channel] = now
    return True


def save_unknown_crop(frame: np.ndarray, box: list, unknown_dir: Path) -> None:
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    unknown_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex[:6]
    path = unknown_dir / f"{_ts()}_{uid}.jpg"
    cv2.imwrite(str(path), crop)
    log.debug("Unknown crop saved → %s", path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-recognize", action="store_true")
    parser.add_argument("--channels", nargs="+", type=int, default=config.CHANNELS)
    args = parser.parse_args()

    recognizer = None
    if not args.no_recognize:
        db_path = Path(config.FACE_DB_PATH)
        if db_path.exists():
            from recognizer import FaceRecognizer
            recognizer = FaceRecognizer(config.FACE_DB_PATH, config.FACE_MATCH_THRESHOLD)
        else:
            log.warning("No face_db.pkl found — running without face recognition. "
                        "Run 'python enroll.py build' to create it.")

    detector = Detector(config.MODEL, config.CONFIDENCE, config.ALERT_CLASSES)
    snap_dir = Path(config.SNAPSHOT_DIR)
    unknown_dir = Path(config.UNKNOWN_FACES_DIR)
    last_alert: dict[int, float] = {}

    streams = {ch: CameraStream(config.rtsp_url(ch), ch).start() for ch in args.channels}
    log.info("Started %d stream(s): channels %s", len(streams), args.channels)

    annotated: dict[int, np.ndarray] = {}
    placeholder = np.zeros((config.GRID_CELL_H, config.GRID_CELL_W, 3), dtype=np.uint8)

    try:
        while True:
            for ch, stream in streams.items():
                frame = stream.read(timeout=0.05)
                if frame is None:
                    annotated.setdefault(ch, placeholder)
                    continue

                ann, detections = detector.run(frame, recognizer)
                annotated[ch] = ann

                alert_detections = _filter_alerts(detections, recognizer)
                if alert_detections:
                    label = ", ".join(_det_label(d) for d in alert_detections)
                    save_snapshot(ch, ann, label, last_alert, config.ALERT_COOLDOWN_SEC, snap_dir)
                    if recognizer:
                        for det in alert_detections:
                            if det["identity"] == "unknown":
                                save_unknown_crop(frame, det["box"], unknown_dir)
                else:
                    for det in detections:
                        if det.get("identity") and det["identity"] not in ("unknown", "no_face"):
                            log.debug("[ch%d] recognized: %s", ch, det["identity"])

            if not args.no_display and annotated:
                grid = build_grid(annotated, config.GRID_CELL_W, config.GRID_CELL_H)
                cv2.imshow("CP Plus — CV Monitor", grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        for stream in streams.values():
            stream.stop()
        cv2.destroyAllWindows()
        log.info("Stopped.")


def _filter_alerts(detections: list, recognizer) -> list:
    """Return detections that should trigger an alert."""
    if not recognizer:
        return detections  # no recognition → alert on everything as before
    alerts = []
    for det in detections:
        identity = det.get("identity")
        if identity in ("unknown", "no_face"):
            alerts.append(det)
        # known name → suppress
    return alerts


def _det_label(det: dict) -> str:
    identity = det.get("identity")
    return identity if identity else det["class_name"]


if __name__ == "__main__":
    main()
