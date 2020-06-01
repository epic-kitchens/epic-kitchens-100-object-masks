import argparse
import re
from pathlib import Path
from typing import Any, Dict, List
from pycocotools.mask import decode as coco_mask_decode

import numpy as np
import pandas as pd
from epic_kitchens.masks.io import save_detections
from epic_kitchens.masks.types import BBox, FrameObjectDetections, ObjectDetection

parser = argparse.ArgumentParser(
    description="Convert raw masks (list of dictionaries) to more data efficient protobuf format for release",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "raw_masks_pkl",
    type=Path,
    help="Path to pickle containing list of raw mask detections",
)
parser.add_argument(
    "releasable_masks_pkl",
    type=Path,
    help="Path to pickle containing list of releasable mask detections",
)
parser.add_argument("--width", type=int, default=456)
parser.add_argument("--height", type=int, default=256)


def get_frame_number(frame_filename: str) -> int:
    return int(re.match(r".*?(\d+)$", frame_filename).group(1))


class Converter:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def convert_video_detections(
        self, video_id: str, raw_masks: Dict[str, Dict[str, Any]]
    ) -> List[FrameObjectDetections]:
        return [
            self.convert_frame_detections(video_id, frame_filename, raw_frame_masks)
            for frame_filename, raw_frame_masks in raw_masks.items()
        ]

    def convert_frame_detections(
        self, video_id: str, frame_filename: str, raw_frame_masks: Dict[str, Any]
    ) -> FrameObjectDetections:
        frame_number = get_frame_number(frame_filename)
        return FrameObjectDetections(
            video_id=video_id,
            frame_number=frame_number,
            objects=[
                self.convert_object_detection(bbox, mask, cls, score)
                for bbox, cls, mask, score in zip(
                    raw_frame_masks["bboxes"],
                    raw_frame_masks["ids"],
                    raw_frame_masks["masks"],
                    raw_frame_masks["scores"],
                )
            ],
        )

    def convert_object_detection(
        self, bbox: np.ndarray, mask: Dict[str, Any], cls: int, score: float
    ) -> ObjectDetection:
        assert bbox.ndim == 1
        assert 'counts' in mask
        assert 'size' in mask
        assert mask['size'][0] == 100
        assert mask['size'][1] == 100
        return ObjectDetection(
            score=score,
            pred_class=cls,
            mask=coco_mask_decode(mask),
            bbox=BBox(
                top_left_x=bbox[0] / self.width,
                top_left_y=bbox[1] / self.height,
                bottom_right_x=bbox[2] / self.width,
                bottom_right_y=bbox[3] / self.height,
            ),
        )


def main(args):
    raw_masks: Dict[str, Dict[str, Any]] = pd.read_pickle(args.raw_masks_pkl)
    video_id = args.raw_masks_pkl.stem
    converter = Converter(height=args.height, width=args.width)
    releasable_masks = converter.convert_video_detections(video_id, raw_masks)
    save_detections(args.releasable_masks_pkl, releasable_masks)


if __name__ == "__main__":
    main(parser.parse_args())
