from typing import Iterator

from epic_kitchens.masks.types import BBox, FrameObjectDetections, ObjectDetection
from epic_kitchens.masks.io import load_detections
from epic_kitchens.masks.coco import class_names
import numpy as np
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Sanity check masks",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("detections_pkl", type=Path, help="Path to masks pkl.")
parser.add_argument(
    "-n", "--n-frames", type=int, help="Expected number of frames in video."
)


class DetectionChecker:
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def check(self, video_detections: Iterator[FrameObjectDetections]) -> None:
        n_detections = 0
        for frame_detections in video_detections:
            n_detections += 1
            self.check_frame_detections(frame_detections)

        if self.n_frames is not None:
            if n_detections != self.n_frames:
                raise ValueError(
                        f"Expected video_detections to contain {self.n_frames}"
                        f"detections, but contained {n_detections}."
                )

    def check_frame_detections(self, frame_detections: FrameObjectDetections) -> None:
        if self.n_frames is not None:
            if not (1 <= frame_detections.frame_number <= self.n_frames):
                raise ValueError(
                    "Expected frame_detections to have frame_number"
                    f"between 1 and {self.n_frames}, but was"
                    f"{frame_detections.frame_number}"
                )
        for obj in frame_detections.objects:
            self.check_object_detection(obj)

    def check_object_detection(self, object_detection: ObjectDetection) -> None:
        self.check_bbox(object_detection.bbox)
        self.check_score(object_detection.score)
        self.check_masks(object_detection.mask)
        self.check_class_id(object_detection.pred_class)

    def check_score(self, score: float) -> None:
        if not (0 <= score <= 1):
            raise ValueError(f"Expected score to be between 0--1 but was {score}")

    def check_bbox(self, bbox: BBox) -> None:
        for coord in [
            "top_left_x",
            "top_left_y",
            "bottom_right_x",
            "bottom_right_y",
        ]:
            value = getattr(bbox, coord)
            if not (0 <= value <= 1):
                raise ValueError(f"Expected bbox {coord} ({value}) to be between 0--1.")
        if not (bbox.top_left_x <= bbox.bottom_right_x):
            raise ValueError(
                f"Expected bbox top_left_x ({bbox.top_left_x}) to be "
                f"less than or equal to bottom_right_x ({bbox.bottom_right_x}"
            )

        if not (bbox.top_left_y <= bbox.bottom_right_y):
            raise ValueError(
                f"Expected bbox top_left_y ({bbox.top_left_y}) to be "
                f"less than or equal to bottom_right_y ({bbox.bottom_right_y}"
            )

    def check_masks(self, mask: np.ndarray) -> None:
        if not isinstance(mask, np.ndarray):
            raise ValueError(
                f"Expected mask to be an instance of np.ndarray but was {type(mask)}"
            )
        if not mask.dtype == np.uint8:
            raise ValueError(f"Expected mask to be of type uint8 but was {mask.dtype}")

    def check_class_id(self, class_id: int) -> None:
        if not (0 <= class_id < len(class_names)):
            raise ValueError(
                f"Expected class id to be between 0--{len(class_names) - 1} but was"
                f" {class_id}"
            )


def main(args):
    detections: Iterator[FrameObjectDetections] = load_detections(args.detections_pkl)
    checker = DetectionChecker(args.n_frames)
    checker.check(detections)


if __name__ == "__main__":
    main(parser.parse_args())
