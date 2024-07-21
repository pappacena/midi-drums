import time
import cv2


class ImageSource:
    def __init__(self, cap) -> None:
        self.cap = cap
        self.stop_on_null_frame = False
        self.frame_wait_ms = 10
    
    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

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