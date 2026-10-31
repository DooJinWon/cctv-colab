"""Only explicitly reviewed intervals may reveal one eligible face per frame."""

from __future__ import annotations
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
from .geometry import clip_box, expand


@dataclass
class EditPlan:
    mode: str = "all"
    style: str = "mosaic"
    face_margin: float = 0.35
    mosaic_block_pixels: int = 18
    keep_audio: bool = False
    reveal_intervals: list = field(default_factory=list)
    manual_masks: list = field(default_factory=list)
    # Bind edits to a particular immutable analysis project.
    analysis_id: str = ""

    def validate(self, frame_count=None):
        if self.mode not in ("all", "selected") or self.style not in ("mosaic", "blur", "solid"):
            raise ValueError("Unknown mode or style.")
        if not 0.1 <= self.face_margin <= 2 or not 8 <= self.mosaic_block_pixels <= 128:
            raise ValueError("Invalid masking settings.")
        for item in self.reveal_intervals:
            start, end = item["start_frame"], item["end_frame"]
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                raise ValueError("Invalid reveal interval.")
            if not item.get("segment_id") or not item.get("confirmed_same_person"):
                raise ValueError(
                    "Reveal intervals require an explicit reviewed-person confirmation."
                )
            if frame_count is not None and end >= frame_count:
                raise ValueError("Reveal interval extends beyond the video.")
        intervals = sorted(self.reveal_intervals, key=lambda x: x["start_frame"])
        for i, a in enumerate(intervals):
            for b in intervals[i + 1 :]:
                if b["start_frame"] > a["end_frame"]:
                    break
                if a["segment_id"] != b["segment_id"]:
                    raise ValueError(
                        "Two different people cannot be revealed in overlapping frame intervals."
                    )
        for item in self.manual_masks:
            keys = item.get("keyframes", [])
            if not keys:
                raise ValueError("Manual mask needs at least one keyframe.")
            frames = [k["frame"] for k in keys]
            if frames != sorted(set(frames)) or not all(
                isinstance(f, int) and f >= 0 for f in frames
            ):
                raise ValueError(
                    "Manual keyframes must be unique, sorted, non-negative frame indices."
                )
            if frame_count is not None and frames[-1] >= frame_count:
                raise ValueError("Manual mask extends beyond the video.")
            for key in keys:
                b = key["box"]
                if len(b) != 4 or not all(math.isfinite(float(v)) and 0 <= v <= 1 for v in b):
                    raise ValueError("Manual boxes must contain normalized coordinates in [0,1].")
                if b[0] >= b[2] or b[1] >= b[3]:
                    raise ValueError("Manual mask has an empty box.")

    def save(self, path):
        self.validate()
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        plan = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        plan.validate()
        return plan


def manual_box(mask, index, width, height):
    keys = mask["keyframes"]
    if not keys[0]["frame"] <= index <= keys[-1]["frame"]:
        return None
    for a, b in zip(keys, keys[1:]):
        if a["frame"] <= index <= b["frame"]:
            fraction = (index - a["frame"]) / (b["frame"] - a["frame"])
            box = np.array(a["box"]) * (1 - fraction) + np.array(b["box"]) * fraction
            return clip_box(box * np.array([width, height, width, height]), width, height)
    return clip_box(np.array(keys[0]["box"]) * [width, height, width, height], width, height)


def mask_boxes(record, plan, width, height):
    """Return only masking boxes. Never restore a target patch over other masks."""
    index = record["index"]
    approved = set()
    if plan.mode == "selected":
        approved = {
            i["segment_id"]
            for i in plan.reveal_intervals
            if i.get("confirmed_same_person") and i["start_frame"] <= index <= i["end_frame"]
        }
    # Defensive check even if a caller bypassed EditPlan.validate().
    if len(approved) > 1:
        approved = set()
    masks = []
    for face in record["faces"]:
        reveal = (
            face.get("reveal_eligible", False)
            and not face.get("ambiguous", False)
            and face.get("segment_id") in approved
        )
        if not reveal:
            box = expand(face["box"], width, height, plan.face_margin)
            if box:
                masks.append(box)
    # Head location cannot safely be guessed from the upper part of a lying body.
    # Use the whole detected person whenever no face is unambiguously associated.
    for person in record["people"]:
        if not person.get("face_count", 0):
            box = expand(person["box"], width, height, 0.08)
            if box:
                masks.append(box)
    for held in record.get("held_boxes", []):
        box = expand(held, width, height, max(0.5, plan.face_margin))
        if box:
            masks.append(box)
    # Manual masks always take precedence, including over an approved target face.
    for mask in plan.manual_masks:
        box = manual_box(mask, index, width, height)
        if box:
            masks.append(box)
    return masks
