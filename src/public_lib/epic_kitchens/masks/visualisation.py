import random
from typing import List, Tuple

import PIL.Image
import numpy as np
from epic_kitchens.masks.types import BBox, FrameObjectDetections
from epic_kitchens.masks._maskrcnn_visualise import display_instances, random_colors
from epic_kitchens.masks.coco import class_names as coco_class_names


def resize_mask(
    mask: np.ndarray, height: int, width: int, smooth: bool = True
) -> np.ndarray:
    assert mask.ndim == 2
    if smooth:
        # The original masks seem to be
        mask_img = PIL.Image.fromarray(mask * 255)
        return (
            np.asarray(
                mask_img.resize((50, 50), PIL.Image.LANCZOS).resize(
                    (width, height), PIL.Image.LANCZOS
                )
            )
            > 128
        ).astype(np.uint8)
    return np.asarray(
        PIL.Image.fromarray(mask).resize((width, height), PIL.Image.NEAREST)
    )


def resize_bbox(bbox: BBox, height: int, width: int) -> Tuple[int, int, int, int]:
    return (
        round(bbox.left * width),
        round(bbox.top * height),
        round(bbox.right * width),
        round(bbox.bottom * height),
    )


class DetectionRenderer:
    def __init__(
        self,
        display_mask: bool = True,
        display_bbox: bool = True,
        score_threshold: float = 0,
        smooth_mask: bool = True,
    ):
        self.display_mask = display_mask
        self.display_bbox = display_bbox
        self.score_threshold = score_threshold
        self.smooth_mask = smooth_mask
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
                masks.append(
                    resize_mask(
                        obj.mask, img.height, img.width, smooth=self.smooth_mask
                    )
                )
                bboxes.append(resize_bbox(obj.bbox, img.height, img.width))
                scores.append(obj.score)

        if len(masks) > 0:
            np_masks = np.stack(masks, axis=-1)
        else:
            np_masks = np.zeros((img.height, img.width, 0), dtype=np.uint8)
        return display_instances(
            np.asarray(img),
            np.array(bboxes),
            np_masks,
            np.array(pred_classes),
            np.array(coco_class_names),
            scores=scores,
            show_mask=self.display_mask,
            show_bbox=self.display_bbox,
            colors=self.colors,
        )
