import cv2
import numpy as np
from ultralytics import YOLO

_IDENTITY_COLORS = {
    "unknown": (0, 0, 255),    # red
    "no_face": (0, 165, 255),  # orange
}
_KNOWN_COLOR = (0, 255, 0)     # green


class Detector:
    def __init__(self, model_path: str, conf: float, alert_classes: list[int] | None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.alert_classes = alert_classes  # None = all COCO classes

    def run(self, frame: np.ndarray, recognizer=None):
        """Return (annotated_frame, detections).

        Each detection dict: {class_id, class_name, confidence, box, identity}.
        identity is None when no recognizer is provided.
        """
        results = self.model(frame, conf=self.conf, classes=self.alert_classes, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

            identity = None
            if recognizer and class_id == 0:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    identity = recognizer.identify(crop)

            detections.append({
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": float(box.conf),
                "box": [x1, y1, x2, y2],
                "identity": identity,
            })

        ann = result.plot()

        for det in detections:
            if det["identity"] is None:
                continue
            x1, y1 = det["box"][0], det["box"][1]
            label = det["identity"]
            color = _IDENTITY_COLORS.get(label, _KNOWN_COLOR)
            cv2.putText(ann, label, (x1, max(y1 - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        return ann, detections
