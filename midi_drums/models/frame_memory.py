import time
from threading import Lock, Thread

import numpy
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ResNetForImageClassification

def sleep(seconds):
    time.sleep(seconds / 2)


class FramesMemory:
    """Memory of the last frames seen, and their embeddings"""
    MAX_FRAMES = 30

    def __init__(
        self, 
        device,
        embeddings_model = "microsoft/resnet-18",
    ):
        self.frame_lock = Lock()
        self.device = device
        self.frames = []
        self.embeddings = []
        self.model = ResNetForImageClassification.from_pretrained(
            embeddings_model,
            output_hidden_states=True,
            device_map=self.device,
        )
        self.processor = AutoImageProcessor.from_pretrained(embeddings_model)

    def _get_embedding(self, frame):
        inputs = self.processor(frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            return output.hidden_states[-1].flatten()
    
    def add_frame(self, frame):
        with self.frame_lock:
            if len(self.frames) >= self.MAX_FRAMES:
                self.frames.pop(0)
                self.embeddings.pop(0)
            self.frames.append(frame)
            self.embeddings.append(self._get_embedding(frame))
    
    def get_embeddings_copy(self):
        with self.frame_lock:
            return self.embeddings[:]


class FrameMemoryCapture(Thread):
    """Background thread to keep updating the frame memory"""
    def __init__(self, frame_memory, source):
        super().__init__()
        self.frame_memory = frame_memory
        self.source = source
        self.should_stop = False
    
    def run(self):
        for frame in self.source:
            if self.should_stop:
                break
            self.frame_memory.add_frame(frame)
    
    def stop(self):
        self.should_stop = True


class DatasetBuilder:
    def __init__(self, frame_memory, max_symbols=5, device="cpu"):
        self.device = device
        self.frame_memory = frame_memory
        self.samples_per_symbol = 10
        self.max_symbols = max_symbols
        self.inputs = []
        self.expected_outputs = []
        self.hit_sleep = 0.75
    
    def request_symbol(self, symbol: int) -> None:
        # Count to 3 and request the symbol hit
        for i in range(3):
            print(f"Hit the #{symbol} symbol starts in {3 - i}...")
            sleep(1)

        for i in range(self.samples_per_symbol):
            print("NOW!")
            embeddings = self.frame_memory.get_embeddings_copy()
            embeddings = torch.Tensor(
                numpy.array([i.cpu().numpy() for i in embeddings])
            ).flatten().to(self.device)
            output = torch.Tensor(
                [1 if i == symbol else 0 for i in range(self.max_symbols)]
            ).to(self.device)
            self.inputs.append(embeddings)
            self.expected_outputs.append(output)
            sleep(self.hit_sleep)

    def build_dataset(self):
        print(f"Hit the symbol every time you see the message")
        print(f"This is the rythm:")
        sleep(1)
        for i in range(3):
            print("NOW!")
            sleep(self.hit_sleep)
        print("Now, for real!")
        for i in range(self.max_symbols):
            print("We will now request the symbol number", i)
            sleep(1)
            self.request_symbol(i)


class DrumsNet(nn.Module):
    def __init__(self, input_size, output_size=5):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

class DumsNetTrainer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self):
        for epoch in range(50):
            for i, (inputs, expected_output) in enumerate(zip(self.dataset.inputs, self.dataset.expected_outputs)):
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.loss_fn(output, expected_output)
                loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch}, loss {loss.item()}")
