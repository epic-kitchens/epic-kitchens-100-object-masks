import pickle
from pathlib import Path
from typing import List, Union

from epic_kitchens.masks.types import FrameObjectDetections


def load_detections(filepath: Union[Path, str]) -> List[FrameObjectDetections]:
    video_id = filepath.stem
    with open(filepath, "rb") as f:
        return [
            FrameObjectDetections.from_protobuf_str(video_id, pb_str) for pb_str in
            pickle.load(f)
        ]


def save_detections(
    filepath: Union[Path, str], detections: List[FrameObjectDetections]
) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)
    with open(filepath, "wb") as f:
        pickle.dump([det.to_protobuf().SerializeToString() for det in detections], f)
