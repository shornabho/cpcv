import logging
import queue
import subprocess
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


def _probe_dimensions(url: str) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            url,
        ],
        capture_output=True, text=True, timeout=10,
    )
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


class CameraStream:
    """Pipes raw BGR frames out of ffmpeg into a queue.

    Uses ffmpeg directly (same as ffplay) so RTSP auth and transport
    are handled identically — no OpenCV RTSP stack involved.
    """

    def __init__(self, url: str, channel: int, fps: int = 10, maxsize: int = 4):
        self.url = url
        self.channel = channel
        self.fps = fps
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._capture, daemon=True, name=f"stream-ch{channel}"
        )

    def start(self) -> "CameraStream":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture(self) -> None:
        while not self._stop.is_set():
            try:
                w, h = _probe_dimensions(self.url)
            except Exception as e:
                log.warning("[ch%d] ffprobe failed (%s), retrying in 5s", self.channel, e)
                time.sleep(5)
                continue

            cmd = [
                "ffmpeg",
                "-rtsp_transport", "tcp",
                "-i", self.url,
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-vf", f"fps={self.fps}",
                "-an",
                "pipe:1",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            frame_bytes = w * h * 3
            log.info("[ch%d] connected (%dx%d @ %dfps)", self.channel, w, h, self.fps)

            try:
                while not self._stop.is_set():
                    raw = proc.stdout.read(frame_bytes)
                    if len(raw) != frame_bytes:
                        log.warning("[ch%d] stream lost, reconnecting", self.channel)
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                    if self._q.full():
                        try:
                            self._q.get_nowait()
                        except queue.Empty:
                            pass
                    self._q.put(frame)
            finally:
                proc.kill()
                proc.wait()

            if not self._stop.is_set():
                time.sleep(2)
