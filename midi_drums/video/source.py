import time

from PIL import Image
import cv2


class ImageSource:
    def __init__(self, cap) -> None:
        self.cap = cap
        self.stop_on_null_frame = False
        self.frame_wait_ms = 5
        self.frames_as_pil = False
    
    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.frames_as_pil:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        return frame
    
    def set_frame_size(self, width: int, height: int) -> None:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def set_fps(self, fps: int) -> None:
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def __iter__(self):
        while True:
            frame = self.get_next_frame()
            if frame is None:
                if self.stop_on_null_frame:
                    break
                else:
                    time.sleep(self.frame_wait_ms / 1000)
                    continue
            yield frame

    def close(self):
        self.cap.release()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoFile(ImageSource):
    def __init__(self, file_path) -> None:
        cap = cv2.VideoCapture(file_path)
        super().__init__(cap)


class Camera(ImageSource):
    def __init__(self, device_id) -> None:
        cap = cv2.VideoCapture(device_id)
        super().__init__(cap)