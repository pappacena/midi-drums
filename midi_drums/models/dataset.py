from io import BytesIO
import json
import random
import re
import time
import logging

from PIL import Image
import ollama

from midi_drums.video.source import ImageSource, frame_to_pil_image

LOGGER = logging.getLogger(__name__)


class DatasetManager:
    """
    Class that gets random frames and create a dataset to train a small model.
    """
    def __init__(
        self, 
        source: ImageSource, 
        ollama_client: ollama.Client, 
        n_samples: int
    ) -> None:
        self.source = source
        self.n_samples = n_samples
        self.samples = []
        self.drumstick_positions = []
        self.ollama_client = ollama_client
        self.json_md_re = re.compile(r'.*```json\n(.+)\n```.*', re.DOTALL)

    def _get_json_from_llm_text(self, text: str) -> dict:
        """
        Get json from text.
        """
        match = self.json_md_re.match(text)
        if match is None:
            return None
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            LOGGER.error(f"Could not parse json from text: {text}")
            return None
        
        if isinstance(data, dict):
            # Sometimes, it comes as {"stick1": position, "stick2": position}
            data = list(data.values())
        
        if not isinstance(data, list):
            LOGGER.error(f"Expected list, got {type(data)}")
            return None
        
        if not len(data):
            LOGGER.error(f"Expected list of length 1 at least, got {len(data)}")
            return None
        
        if isinstance(data[0], list):
            # Convert lists to dicts
            return [
                dict(x1=x1, x2=x2, y1=y1, y2=y2)
                for x1, x2, y1, y2 in data
            ]
    
        for item in data:
            if not all(i in item for i in ('x1', 'x2', 'y1', 'y2')):
                LOGGER.error(f"Expected dict with keys x1, x2, y1, y2, got {data}")
                return None
        return data
        
    
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
        for frame in self.samples:
            buff = BytesIO()
            frame_to_pil_image(frame).save(buff, format='PNG')
            buff.seek(0)
            resp = self.ollama_client.chat(
                model='llava', 
                messages=[
                    dict(
                        role='user', 
                        content=(
                            'Output only json list of 2 dicts with format '
                            '[{x1,y1,x2,y2}, {x1,y1,x2,y2}]. '
                            'No explanation: give me the position of the boxes surounding the drum sticks '
                            'in the image, in percentage of image size.'
                        ),
                        images=[buff]
                    )
                ]
            )
            response = resp["message"]["content"]
            self.drumstick_positions.append(self._get_json_from_llm_text(response))