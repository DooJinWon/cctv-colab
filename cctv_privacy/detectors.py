"""Pretrained YOLO11 person boxes + YuNet faces; no identity database or training."""

from __future__ import annotations
import hashlib
import math
import os
import requests
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from .geometry import clip_box, conservative_merge, inverse_rotated_box

YUNET_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
)


def download_yunet(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        temp = path.with_suffix(".download")
        errors = []
        for url in YUNET_URLS:
            try:
                with (
                    requests.get(url, stream=True, timeout=(15, 60)) as src,
                    temp.open("wb") as dst,
                ):
                    src.raise_for_status()
                    for chunk in src.iter_content(1024 * 1024):
                        if chunk:
                            dst.write(chunk)
                if temp.stat().st_size < 100_000:
                    raise RuntimeError(
                        "YuNet 다운로드가 모델 대신 오류 페이지 또는 Git LFS 포인터를 반환했습니다."
                    )
                temp.replace(path)
                break
            except (requests.RequestException, RuntimeError) as exc:
                errors.append(f"{url}: {exc}")
            finally:
                temp.unlink(missing_ok=True)
        if not path.exists():
            raise RuntimeError("YuNet 다운로드 실패:\n" + "\n".join(errors))
    return path


def tile_origins(length, size, overlap):
    if size >= length:
        return [0]
    step = max(1, round(size * (1 - overlap)))
    return sorted(set(list(range(0, length - size + 1, step)) + [length - size]))


class FaceDetector:
    def __init__(
        self, model_path, threshold=0.55, rotations=(0, 90, 180, 270), max_side=1280, tile_size=960
    ):
        if not rotations or any(r not in (0, 90, 180, 270) for r in rotations):
            raise ValueError("rotations must contain 0, 90, 180, or 270 degrees.")
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), threshold, 0.3, 5000
        )
        self.rotations, self.max_side, self.tile_size = tuple(rotations), max_side, tile_size

    def _infer(self, image, offset=(0, 0)):
        height, width = image.shape[:2]
        detections = []
        for rotation in self.rotations:
            k = rotation // 90
            rotated = np.ascontiguousarray(np.rot90(image, k))
            rh, rw = rotated.shape[:2]
            scale = min(1.0, self.max_side / max(rh, rw))
            iw, ih = max(32, round(rw * scale)), max(32, round(rh * scale))
            prepared = cv2.resize(rotated, (iw, ih)) if (iw, ih) != (rw, rh) else rotated
            self.detector.setInputSize((iw, ih))
            _, faces = self.detector.detect(prepared)
            if faces is None:
                continue
            for face in faces:
                # Rounding can make scale_x and scale_y slightly different.
                x, y, w, h = map(float, face[:4])
                box = inverse_rotated_box(
                    [x * rw / iw, y * rh / ih, (x + w) * rw / iw, (y + h) * rh / ih],
                    k,
                    width,
                    height,
                )
                box = [
                    box[0] + offset[0],
                    box[1] + offset[1],
                    box[2] + offset[0],
                    box[3] + offset[1],
                ]
                detections.append({"box": box, "score": float(face[-1])})
        return detections

    def detect(self, frame):
        height, width = frame.shape[:2]
        items = self._infer(frame)
        if self.tile_size and max(height, width) > self.tile_size:
            for y in tile_origins(height, self.tile_size, 0.25):
                for x in tile_origins(width, self.tile_size, 0.25):
                    crop = frame[
                        y : min(y + self.tile_size, height), x : min(x + self.tile_size, width)
                    ]
                    items.extend(self._infer(crop, (x, y)))
        valid = []
        for item in items:
            box = clip_box(item["box"], width, height)
            if box:
                valid.append({"box": box, "score": item["score"]})
        return conservative_merge(valid)


class PersonDetectorTracker:
    def __init__(self, model_path, fps, device="auto", image_size=960, threshold=0.1):
        # Keep third-party caches in the task directory, not a user's home directory.
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker
        import torch

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device, self.image_size, self.threshold = device, image_size, threshold
        self.model = YOLO(str(model_path))
        self.tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                track_buffer=30,
                match_thresh=0.8,
                fuse_score=True,
            ),
            frame_rate=round(fps),
        )
        self.previous = {}
        self.shot = 0
        self.serial = 0

    def reset(self):
        self.tracker.reset()
        self.previous.clear()
        self.shot += 1

    def detect(self, frame, index):
        result = self.model.predict(
            frame,
            classes=[0],
            conf=self.threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes.cpu().numpy()
        # Retain ALL raw detections, including those without a confirmed tracking ID.
        height, width = frame.shape[:2]
        people = []
        for box, score in zip(boxes.xyxy, boxes.conf):
            people.append(
                {
                    "box": clip_box(box, width, height),
                    "score": float(score),
                    "track_id": None,
                    "segment_id": None,
                    "stable_frames": 0,
                }
            )
        tracks = self.tracker.update(boxes, img=frame)
        # ByteTrack's returned detection index can refer to a filtered confidence
        # subset. Associate geometrically instead of indexing the raw list with it.
        from scipy.optimize import linear_sum_assignment
        from .geometry import iou

        matches = []
        if len(tracks) and people:
            overlaps = np.array(
                [[iou(t[:4], p["box"]) if p["box"] else 0 for p in people] for t in tracks]
            )
            rows, cols = linear_sum_assignment(-overlaps)
            for row, col in zip(rows, cols):
                alternatives = np.delete(overlaps[row], col)
                if overlaps[row, col] < 0.30 or (
                    len(alternatives) and overlaps[row, col] - max(alternatives) < 0.05
                ):
                    continue
                matches.append((tracks[row], int(col)))
        for track, detection_index in matches:
            person = people[detection_index]
            local_id = int(track[4])
            person["track_id"] = f"{self.shot}:{local_id}"
            previous = self.previous.get(local_id)
            gap = previous is None or previous["index"] != index - 1
            jumped = False
            if previous and not gap and person["box"]:
                a, b = person["box"], previous["box"]
                distance = math.hypot(
                    (a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2
                )
                jumped = distance > 0.65 * max(1, math.hypot(b[2] - b[0], b[3] - b[1]))
            if gap or jumped:
                self.serial += 1
                sid = f"segment-{self.serial:06d}"
                stable = 1
            else:
                sid = previous["segment_id"]
                stable = previous["stable_frames"] + 1
            person.update(
                segment_id=sid,
                stable_frames=stable,
                reacquired=bool(previous and gap),
                jumped=jumped,
            )
            self.previous[local_id] = {
                "index": index,
                "box": person["box"],
                "segment_id": sid,
                "stable_frames": stable,
            }
        # Bound bookkeeping independently of video length.
        self.previous = {k: v for k, v in self.previous.items() if index - v["index"] <= 120}
        return [p for p in people if p["box"]]


def model_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
