from epic_kitchens.masks.io import load_detections
import os
import numpy as np
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Sanity check masks",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--detections_pkls", type=Path, help="Path to masks pkl."
)
parser.add_argument(
    "--frames", type=Path, help="Path to rgb frames."
)


class DetectionChecker:
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def check(self, video_detections) -> None:
        if self.n_frames is not None:
            if len(video_detections) != self.n_frames:
                raise ValueError(
                    f"Expected video_detections to contain {self.n_frames}"
                    f"detections, but contained {len(video_detections)}."
                )
        for frame_detections in video_detections:
            self.check_frame_detections(frame_detections)

    def check_frame_detections(self, frame_detections) -> None:
        if self.n_frames is not None:
            if not (1 <= frame_detections.frame_number <= self.n_frames):
                raise ValueError(
                    "Expected frame_detections to have frame_number"
                    f"between 1 and {self.n_frames}, but was"
                    f"{frame_detections.frame_number}"
                )
        for obj in frame_detections.objects:
            self.check_object_detection(obj)


    def check_object_detection(self, object_detection) -> None:
        self.check_bbox(object_detection.bbox)
        self.check_score(object_detection.score)
        self.check_masks(object_detection.mask)
        self.check_predicted_id(object_detection.pred_class)


    def check_score(self, score) -> None:
        if not (0 <= score <= 1):
            raise ValueError(f"Expected score to be between 0--1 but was {score}")

    def check_bbox(self, bbox) -> None:
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

    def check_masks(self, mask) -> None:
        if not isinstance(mask, np.ndarray):
            raise ValueError(
                f"Expected hand state to be an instance of Numpy but "
                f"was {mask.state}"
            )

    def check_predicted_id(self, pre_id) -> None:
        if not (0 <= pre_id <= 80):
            raise ValueError(
                f"Expected predicted class to be between 0--81 but was {pre_id}"
            )


def main(args):
    pids = sorted(os.listdir(args.detections_pkls))
    for pid in pids:
        pid_path = os.path.join(args.detections_pkls, pid)
        vids = sorted(os.listdir(pid_path))
        for vid in vids:
            print(vid)
            vid_path = os.path.join(pid_path, vid)
            frame_path = os.path.join(args.frames, pid, vid.split('.')[0])
            n_frames = len(os.listdir(frame_path))
            MasksInfo = load_detections(vid_path)
            checker = DetectionChecker(n_frames)
            checker.check(MasksInfo)


if __name__ == "__main__":
    main(parser.parse_args())




