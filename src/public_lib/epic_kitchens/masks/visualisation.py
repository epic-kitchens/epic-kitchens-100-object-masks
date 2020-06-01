import random
from typing import List, Tuple

import PIL.Image
import numpy as np
from .types import BBox, FrameObjectDetections
from ._maskrcnn_visualise import display_instances, random_colors
from .coco import class_names as coco_class_names


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    assert mask.ndim == 2
    mask_img = PIL.Image.fromarray(mask)
    return np.asarray(mask_img.resize((width, height), PIL.Image.LANCZOS))


def resize_bbox(bbox: BBox, height: int, width: int) -> Tuple[int, int, int, int]:
    return (
        round(bbox.top_left_x * width),
        round(bbox.top_left_y * height),
        round(bbox.bottom_right_x * width),
        round(bbox.bottom_right_y * height),
    )


class DetectionRenderer:
    def __init__(self, display_mask: bool = True, display_bbox: bool = True,
                 score_threshold: float = 0):
        self.display_mask = display_mask
        self.display_bbox = display_bbox
        self.score_threshold = score_threshold
        original_random_state = random.getstate()
        # Ensure colors are consistent across instances of detection renderer.
        random.seed(42)
        self.colors = random_colors(len(coco_class_names))
        random.setstate(original_random_state)

    def render_detections(
        self, img: PIL.Image.Image, detection: FrameObjectDetections
    ) -> PIL.Image.Image:
        img = img.copy()
        bboxes: List[Tuple[int, int, int, int]] = []
        masks: [np.ndarray] = []

        pred_classes = []
        scores = []
        for obj in detection.objects:
            if obj.score > self.score_threshold:
                pred_classes.append(obj.pred_class)
                masks.append(resize_mask(obj.mask, img.height, img.width))
                bboxes.append(resize_bbox(obj.bbox, img.height, img.width))
                scores.append(obj.score)

        return display_instances(
            np.asarray(img),
            np.array(bboxes),
            np.stack(masks, axis=-1),
            np.array(pred_classes),
            np.array(coco_class_names),
            scores=scores,
            show_mask=self.display_mask,
            show_bbox=self.display_bbox,
            colors=self.colors,
        )
