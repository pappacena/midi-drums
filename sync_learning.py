import time
import torch

from midi_drums.models.frame_memory import DatasetBuilder, DrumsNet, DumsNetTrainer, FrameMemoryCapture, FramesMemory
from midi_drums.video.source import Camera


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_EMBEDDINGS_SIZE = 25088  # Hardcoded for resnet-18


cur_sec = prev_sec = int(time.time())
frames = FramesMemory(
    device=DEVICE, 
    embeddings_model="microsoft/resnet-18"
)

camera = Camera(2)
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

    model = DrumsNet(
        input_size=len(dataset.inputs[0]),
        output_size=len(dataset.expected_outputs[0]),
    ).to(DEVICE)

    DumsNetTrainer(model, dataset).train()

    import ipdb; ipdb.set_trace()
    print(model(dataset.inputs[0]))
finally:
    frame_capture.stop()
    frame_capture.join()
    camera.close()
