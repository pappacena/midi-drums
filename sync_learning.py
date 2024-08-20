import time
import torch

from midi_drums.models.frame_memory import DatasetBuilder, DrumsNet, DumsNetTrainer, FrameMemoryCapture, FramesMemory
from midi_drums.video.source import Camera


CAMERA_ID = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_EMBEDDINGS_SIZE = 25088  # Hardcoded for resnet-18


frames = FramesMemory(
    device=DEVICE, 
    embeddings_model="microsoft/resnet-18"
)

camera = Camera(CAMERA_ID)
camera.set_frame_size(640, 480)
camera.set_fps(30)

frame_capture = FrameMemoryCapture(frames, camera)
frame_capture.start()

try:
    dataset = DatasetBuilder(
        frame_memory=frames, 
        max_symbols=3,
        device=DEVICE,
    )
    dataset.build_dataset()
finally:
    frame_capture.stop()
    frame_capture.join()
    camera.close()

# Train the model
model = DrumsNet(
    input_size=len(dataset.inputs[0]),
    output_size=len(dataset.expected_outputs[0]),
).to(DEVICE)

DumsNetTrainer(model, dataset).train()

import ipdb; ipdb.set_trace()
print(model(dataset.inputs[0]))

# Use the model
with Camera(CAMERA_ID) as camera:
    for frame in camera:
        frames.add_frame(frame)
        embeddings = torch.stack(
            frames.get_embeddings_copy()
        ).flatten().to(DEVICE)
        output = model(embeddings)
        print(torch.argmax(output), output)
