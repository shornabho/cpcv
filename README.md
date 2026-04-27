# cpcv — CP Plus CCTV Smart Monitor

Real-time CV pipeline for CP Plus CCTV systems. Detects persons via YOLOv8 and identifies known family members using ArcFace face recognition — alerting only on strangers.

## How it works

```
ffmpeg (RTSP) → person detection (YOLO) → face recognition (ArcFace)
    → known person  → suppress alert
    → unknown face  → snapshot + save crop for review
    → no face seen  → snapshot (conservative)
```

Streams from all channels are read in parallel threads and displayed in a 2×2 grid.

## Setup

**Requirements:** Python 3.12+, ffmpeg, ffprobe

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root:
```
CAMERA_PASS=your_dvr_password
```

## Enrolling family members

**From existing photos:**
```bash
mkdir -p known_faces/your_name
cp /path/to/photos/*.jpg known_faces/your_name/
python enroll.py build
```

**From CCTV captures** — run the pipeline first to accumulate unknown crops, then:
```bash
python enroll.py review
```
In the review window: type a name → Enter to save, Esc to skip, D to delete.
The face DB rebuilds automatically after each review session.

## Running

```bash
python main.py                   # all 4 channels, display + recognition
python main.py --channels 1 2    # specific channels only
python main.py --no-display      # headless / server mode
python main.py --no-recognize    # disable face recognition, alert on all persons
```

Press **Q** to quit the display window.

## Configuration

All tunable settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `CHANNELS` | `[1,2,3,4]` | Active camera channels |
| `MODEL` | `yolov8n.pt` | YOLO model (`yolov8s.pt` for better accuracy) |
| `CONFIDENCE` | `0.45` | YOLO detection confidence threshold |
| `ALERT_CLASSES` | `[0]` | COCO class IDs to watch (0 = person) |
| `ALERT_COOLDOWN_SEC` | `10` | Min seconds between snapshots per channel |
| `FACE_MATCH_THRESHOLD` | `0.40` | Cosine distance for face match (lower = stricter) |
| `SNAPSHOT_DIR` | `snapshots/` | Where alert frames are saved |
| `UNKNOWN_FACES_DIR` | `unknown_faces/` | Where unrecognised face crops accumulate |
| `LOG_FILE` | `cpcv.log` | Persistent log across runs |

## File layout

```
cpcv/
├── config.py          — all settings
├── stream.py          — threaded ffmpeg RTSP reader, auto-reconnects
├── detector.py        — YOLOv8 wrapper with identity label overlay
├── recognizer.py      — ArcFace embeddings, hot-reloads face DB
├── enroll.py          — build and review face database
├── main.py            — pipeline orchestrator
├── known_faces/       — enrolled persons (gitignored)
│   └── <name>/*.jpg
├── unknown_faces/     — detected stranger crops pending review (gitignored)
├── snapshots/         — alert frames (gitignored)
└── cpcv.log           — persistent log (gitignored)
```

## Notes

- First run downloads YOLOv8 (~6 MB) and ArcFace + RetinaFace models (~500 MB total).
- The face DB (`face_db.pkl`) hot-reloads at runtime — no restart needed after enrollment.
- RTSP streams use ffmpeg directly (not OpenCV) to handle CP Plus Digest authentication.
- Tested on CP Plus CP-UVR-0401E1-IC (4-channel DVR, CIF sub-stream 352×288).
