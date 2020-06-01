import PIL.Image
from .types import FrameObjectDetections


class DetectionRenderer:
    def __init__(self):
        pass

    def render_detection(
        self, img: PIL.Image.Image, detection: FrameObjectDetections
    ) -> PIL.Image.Image:
        pass
