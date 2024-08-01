import random
import logging

import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from midi_drums.video.source import ImageSource, frame_to_pil_image

LOGGER = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetManager:
    """
    Class that gets random frames and create a dataset to train a small model.
    """
    def __init__(
        self, 
        source: ImageSource, 
        n_samples: int
    ) -> None:
        self.source = source
        self.n_samples = n_samples
        self.samples = []
        self.drumstick_positions = []        
    
    def _collect_samples(self) -> list[Image]:
        """
        Collects samples from the source.
        """
        for frame in self.source:
            # Get 10% of the frames
            if random.random() < 0.1:
                continue
            self.samples.append(frame)
            if len(self.samples) >= self.n_samples:
                break

    def _guess_drumstick_positions(self):
        """
        Get drumstick position for each frame.
        """
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble",
            device_map=DEVICE,
            low_cpu_mem_usage=True,
        )
        
        for frame in self.samples:
            image = frame_to_pil_image(frame)
            inputs = processor(
                text=[["the top tip of a drumstick"]], 
                images=image, 
                return_tensors="pt"
            ).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            
            target_sizes = torch.Tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs=outputs, 
                target_sizes=target_sizes,
                threshold=0.1
            )
            i = 0
            boxes, scores = results[i]["boxes"], results[i]["scores"]
            result_boxes = []
            for box, score in zip(boxes, scores):
                box = [round(i, 2) for i in box.tolist()]
                result_boxes.append((score, box))
                LOGGER.info(
                    f"Detected with confidence {round(score.item(), 3)} at location {box}"
                )
            result_boxes = sorted(result_boxes, key=lambda x: x[0], reverse=True)
            if result_boxes:
                self.drumstick_positions.append(
                    # Add only the 2 highest confidence boxes
                    [result_boxes for _, result_boxes in result_boxes[:2]]
                )