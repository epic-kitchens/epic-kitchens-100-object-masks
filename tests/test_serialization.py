from pathlib import Path

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from epic_kitchens.masks.types import FrameObjectDetections, ObjectDetection, BBox
from epic_kitchens.masks.io import load_detections, save_detections
from pycocotools.mask import encode as coco_mask_encode


def gen_mask():
    mask = np.zeros((100, 100), dtype=np.uint8, order='F')
    mask[10:30, 20:40] = 1
    return mask


def assert_bbox_close(expected_bbox: BBox, actual_bbox: BBox):
    assert_almost_equal(actual_bbox.top_left_x, expected_bbox.top_left_x)
    assert_almost_equal(actual_bbox.top_left_y, expected_bbox.top_left_y)
    assert_almost_equal(actual_bbox.bottom_right_x, expected_bbox.bottom_right_x)
    assert_almost_equal(actual_bbox.bottom_right_y, expected_bbox.bottom_right_y)


def test_serialisation_round_trip_is_idempotent(tmpdir):
    tmpfile = Path(str(tmpdir / "P01_101.pkl"))

    detections = FrameObjectDetections(
        video_id="P01_101",
        frame_number=10,
        objects=[
            ObjectDetection(
                bbox=BBox(
                    top_left_x=0.1,
                    top_left_y=0.2,
                    bottom_right_x=0.3,
                    bottom_right_y=0.4,
                ),
                _coco_mask_counts=coco_mask_encode(gen_mask())['counts'],
                pred_class=42,
                score=0.8,
            )
        ],
    )
    save_detections(tmpfile, [detections])
    loaded_detections = next(load_detections(tmpfile))

    assert loaded_detections.video_id == detections.video_id
    assert loaded_detections.frame_number == detections.frame_number
    assert len(loaded_detections.objects) == len(detections.objects)

    loaded_obj_det = loaded_detections.objects[0]
    obj_det = detections.objects[0]
    assert loaded_obj_det.pred_class == obj_det.pred_class
    assert_almost_equal(loaded_obj_det.score, obj_det.score)
    assert_array_equal(loaded_obj_det.mask, obj_det.mask)
    assert not (obj_det.mask == 0).all()
    assert_bbox_close(obj_det.bbox, loaded_obj_det.bbox)
