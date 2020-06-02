from typing import List

import numpy as np
from dataclasses import dataclass
from pycocotools.mask import decode as coco_mask_decode

from . import types_pb2 as pb

__all__ = ["BBox", "FrameObjectDetections", "ObjectDetection"]


@dataclass
class BBox:
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float

    @staticmethod
    def from_protobuf(bbox: pb.BBox) -> "BBox":
        return BBox(
            top_left_x=bbox.top_left_x,
            top_left_y=bbox.top_left_y,
            bottom_right_x=bbox.bottom_right_x,
            bottom_right_y=bbox.bottom_right_y,
        )

    def to_protobuf(self) -> pb.BBox:
        bbox = pb.BBox()
        bbox.top_left_x = self.top_left_x
        bbox.top_left_y = self.top_left_y
        bbox.bottom_right_x = self.bottom_right_x
        bbox.bottom_right_y = self.bottom_right_y
        assert bbox.IsInitialized()
        return bbox


@dataclass
class ObjectDetection:
    bbox: BBox
    score: float
    pred_class: int

    _coco_mask_counts: bytes

    @staticmethod
    def from_protobuf(detection: pb.ObjectDetection) -> "ObjectDetection":
        return ObjectDetection(
            bbox=BBox.from_protobuf(detection.bbox),
            score=detection.score,
            pred_class=detection.pred_class,
            _coco_mask_counts=detection.coco_mask,
        )

    def to_protobuf(self) -> pb.ObjectDetection:
        detection = pb.ObjectDetection()
        detection.bbox.MergeFrom(self.bbox.to_protobuf())
        detection.coco_mask = self._coco_mask_counts
        detection.score = self.score
        detection.pred_class = self.pred_class
        assert detection.IsInitialized()
        return detection

    @property
    def mask(self) -> np.ndarray:
        return coco_mask_decode({
            'counts': self._coco_mask_counts,
            'size': [100, 100]
        })


@dataclass
class FrameObjectDetections:
    video_id: str
    frame_number: int
    objects: List[ObjectDetection]

    @staticmethod
    def from_protobuf(
        video_id: str, frame_detections: pb.FrameObjectDetections
    ) -> "FrameObjectDetections":
        return FrameObjectDetections(
            video_id=video_id,
            frame_number=frame_detections.frame_number,
            objects=[
                ObjectDetection.from_protobuf(obj) for obj in frame_detections.objects
            ],
        )

    @staticmethod
    def from_protobuf_str(video_id: str, pb_str: bytes) -> "FrameObjectDetections":
        detections = pb.FrameObjectDetections()
        detections.MergeFromString(pb_str)
        return FrameObjectDetections.from_protobuf(video_id, detections)

    def to_protobuf(self) -> pb.FrameObjectDetections:
        detections = pb.FrameObjectDetections()
        detections.frame_number = self.frame_number
        detections.objects.extend([obj.to_protobuf() for obj in self.objects])
        assert detections.IsInitialized()
        return detections
