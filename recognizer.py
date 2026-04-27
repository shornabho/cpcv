import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FACE_MODEL = "ArcFace"
DETECTOR_BACKEND = "retinaface"  # most reliable; opencv/yunet miss low-res faces
FACE_CONFIDENCE_MIN = 0.3


def _embedding(img: np.ndarray) -> list[float] | None:
    """Return ArcFace embedding for the most prominent face in img, or None."""
    from deepface import DeepFace  # lazy import — model downloads on first call
    try:
        result = DeepFace.represent(
            img,
            model_name=FACE_MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        if not result:
            log.debug("deepface returned empty result")
            return None
        conf = result[0].get("face_confidence", 0)
        log.debug("face_confidence=%.3f (threshold=%.2f)", conf, FACE_CONFIDENCE_MIN)
        if conf < FACE_CONFIDENCE_MIN:
            return None
        return result[0]["embedding"]
    except Exception as e:
        log.debug("embedding failed: %s", e)
        return None


def _cosine_distance(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class FaceRecognizer:
    def __init__(self, db_path: str, threshold: float):
        self.db_path = Path(db_path)
        self.threshold = threshold
        self._db: dict[str, list[list[float]]] = {}
        self._db_mtime: float = 0.0
        self._reload()

    def _reload(self) -> None:
        if not self.db_path.exists():
            return
        mtime = self.db_path.stat().st_mtime
        if mtime > self._db_mtime:
            with open(self.db_path, "rb") as f:
                self._db = pickle.load(f)
            self._db_mtime = mtime
            log.info("Face DB loaded: %d person(s) — %s",
                     len(self._db), ", ".join(self._db.keys()))

    def identify(self, person_crop: np.ndarray) -> str:
        """Returns a known name, 'unknown', or 'no_face'."""
        self._reload()

        emb = _embedding(person_crop)
        if emb is None:
            return "no_face"

        if not self._db:
            return "unknown"

        best_name, best_dist = None, float("inf")
        for name, embeddings in self._db.items():
            for ref in embeddings:
                dist = _cosine_distance(emb, ref)
                if dist < best_dist:
                    best_dist, best_name = dist, name

        return best_name if best_dist < self.threshold else "unknown"
