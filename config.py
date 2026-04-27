import os
from dotenv import load_dotenv

load_dotenv()

CAMERA_HOST = "192.168.31.167"
CAMERA_PORT = 554
CAMERA_USER = "admin"
CAMERA_PASS = os.environ["CAMERA_PASS"]
CHANNELS = [1, 2, 3, 4]

# Detection
MODEL = "yolov8n.pt"       # nano = fast; swap to yolov8s.pt for better accuracy
CONFIDENCE = 0.45
# None = all COCO classes; restrict to e.g. [0, 2] for person + car only
ALERT_CLASSES = [0]        # 0 = person

# Alerting
ALERT_COOLDOWN_SEC = 10    # min seconds between snapshots per channel
SNAPSHOT_DIR = "snapshots"

# Logging
LOG_FILE = "cpcv.log"

# Face recognition
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
FACE_DB_PATH = "face_db.pkl"
FACE_MATCH_THRESHOLD = 0.40   # cosine distance; lower = stricter match

# Display
GRID_CELL_W = 640
GRID_CELL_H = 360


def rtsp_url(channel: int, subtype: int = 1) -> str:
    """subtype=1 is the sub-stream (lower res, faster for inference)."""
    print("Trying:", f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_HOST}:{CAMERA_PORT}/cam/realmonitor?channel={channel}&subtype={subtype}")
    return (
        f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_HOST}:{CAMERA_PORT}"
        f"/cam/realmonitor?channel={channel}&subtype={subtype}"
    )
