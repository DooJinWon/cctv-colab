"""Analysis pass. A successful, complete pass is required before rendering."""

from __future__ import annotations
import csv
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np
from .geometry import area, intersection, iou
from .store import Store
from .video import file_digest


@dataclass
class AnalysisConfig:
    person_model: str = "yolo11n.pt"
    person_image_size: int = 960
    person_threshold: float = 0.10
    face_threshold: float = 0.55
    face_rotations: tuple = (0, 90, 180, 270)
    face_max_side: int = 1280
    face_tile_size: int = 960
    device: str = "auto"
    minimum_stable_frames: int = 3
    crowd_iou: float = 0.10
    hold_seconds: float = 0.20
    cut_difference: float = 0.35

    def validate(self):
        if self.person_image_size < 320 or self.face_max_side < 320:
            raise ValueError("Model input sizes must be at least 320.")
        if self.face_tile_size and self.face_tile_size < 320:
            raise ValueError("Face tile size must be zero or at least 320.")
        for value in (
            self.person_threshold,
            self.face_threshold,
            self.crowd_iou,
            self.cut_difference,
        ):
            if not 0 < value < 1:
                raise ValueError("Thresholds must be between zero and one.")
        if self.minimum_stable_frames < 1 or not 0 <= self.hold_seconds <= 2:
            raise ValueError("Invalid temporal settings.")


def associate(people, faces, config):
    """Spatial association is uncertain; never treat it as biometric identity."""
    flags = []
    for person in people:
        person["face_count"] = 0
    for face in faces:
        candidates = [
            i
            for i, p in enumerate(people)
            if intersection(face["box"], p["box"]) / max(1, area(face["box"])) >= 0.65
        ]
        face["person_index"] = candidates[0] if len(candidates) == 1 else None
        face["ambiguous"] = len(candidates) > 1
        face["reveal_eligible"] = False
        face["segment_id"] = None
        if len(candidates) == 1:
            people[candidates[0]]["face_count"] += 1
        else:
            flags.append("ambiguous_face" if candidates else "unassigned_face")
    for i, person in enumerate(people):
        person["crowded"] = any(
            j != i and iou(person["box"], other["box"]) > config.crowd_iou
            for j, other in enumerate(people)
        )
        if not person["face_count"]:
            flags.append("person_without_face")
        if person["crowded"]:
            flags.append("overlapping_people")
        if not person["segment_id"]:
            flags.append("untracked_person")
        if person.get("reacquired"):
            flags.append("track_reacquired_new_segment")
        if person.get("jumped"):
            flags.append("track_jump_new_segment")
    for face in faces:
        if face["person_index"] is None:
            continue
        person = people[face["person_index"]]
        face["segment_id"] = person["segment_id"]
        face["reveal_eligible"] = bool(
            person["segment_id"]
            and person["stable_frames"] >= config.minimum_stable_frames
            and person["face_count"] == 1
            and not person["crowded"]
        )
        if person["face_count"] != 1:
            face["ambiguous"] = True
            flags.append("multiple_faces_one_person")
    return flags


def analyze(project, config=None, progress=None, detector_pair=None):
    """detector_pair is injectable for deterministic integration tests."""
    project = Path(project).resolve()
    config = config or AnalysisConfig()
    config.validate()
    info = json.loads((project / "manifest.json").read_text())
    database = project / "analysis.sqlite"
    if database.exists():
        raise FileExistsError(
            "Analysis already exists. Use a new project or explicitly remove the incomplete analysis file."
        )
    store = Store(database)
    store.set_meta("complete", False)
    store.set_meta("analysis_id", str(uuid.uuid4()))
    store.set_meta("config", asdict(config))
    store.set_meta("manifest", info)
    prepared_path = project / info["prepared_name"]
    prepared_digest = file_digest(prepared_path)
    store.set_meta("prepared_sha256", prepared_digest)
    store.commit()
    try:
        if detector_pair is None:
            from .detectors import FaceDetector, PersonDetectorTracker, download_yunet, model_digest

            models = project / "models"
            face_model = download_yunet(models)
            people_detector = PersonDetectorTracker(
                models / config.person_model,
                info["fps"],
                config.device,
                config.person_image_size,
                config.person_threshold,
            )
            faces_detector = FaceDetector(
                face_model,
                config.face_threshold,
                config.face_rotations,
                config.face_max_side,
                config.face_tile_size,
            )
            store.set_meta(
                "model_sha256",
                {
                    "yunet": model_digest(face_model),
                    "person": model_digest(models / config.person_model),
                },
            )
            store.set_meta("device", people_detector.device)
        else:
            people_detector, faces_detector = detector_pair
            store.set_meta("device", "injected-test-detectors")
    except BaseException:
        store.close()
        raise
    cap = cv2.VideoCapture(str(project / info["prepared_name"]))
    if not cap.isOpened():
        store.close()
        raise ValueError("Prepared video could not be opened.")
    started = time.perf_counter()
    previous_gray = None
    previous_count = None
    hold = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= info["frame_count"]:
                raise RuntimeError("Unexpected additional video frames.")
            if frame.shape[:2] != (info["height"], info["width"]):
                raise RuntimeError("Video dimensions changed during decoding.")
            gray = (
                cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
                / 255
            )
            cut = (
                previous_gray is not None
                and float(np.mean(np.abs(gray - previous_gray))) > config.cut_difference
            )
            previous_gray = gray
            flags = []
            if cut:
                people_detector.reset()
                hold = []
                flags.append("possible_scene_cut")
            people = people_detector.detect(frame, index)
            faces = faces_detector.detect(frame)
            flags += associate(people, faces, config)
            if not people and not faces:
                flags.append("no_detections")
            if previous_count is not None and len(people) < previous_count:
                flags.append("person_count_drop")
            previous_count = len(people)
            # Keep unmatched previous face areas briefly, always masked (never revealed).
            hold_frames = round(config.hold_seconds * info["fps"])
            active_hold = []
            for item in hold:
                if index - item["last_seen"] <= hold_frames and not any(
                    iou(item["box"], f["box"]) > 0.15 for f in faces
                ):
                    active_hold.append(item)
            if active_hold:
                flags.append("held_face_region")
            record = {
                "index": index,
                "timestamp": index / info["fps"],
                "people": people,
                "faces": faces,
                "held_boxes": [h["box"] for h in active_hold],
                "flags": flags,
            }
            hold = active_hold + [{"box": f["box"], "last_seen": index} for f in faces]
            store.put(record)
            index += 1
            if index % 30 == 0:
                store.commit()
            if progress and (index % 10 == 0 or index == info["frame_count"]):
                progress(index, info["frame_count"])
        if index != info["frame_count"]:
            raise RuntimeError(
                f'Premature decoding end: {index}/{info["frame_count"]}. Rendering is disabled.'
            )
        if file_digest(prepared_path) != prepared_digest:
            raise RuntimeError("Prepared video changed during analysis. Export is disabled.")
        elapsed = time.perf_counter() - started
        store.set_meta("elapsed_seconds", elapsed)
        store.set_meta("complete", True)
        store.commit()
        summary = {
            "frames": index,
            "elapsed_seconds": elapsed,
            "analysis_fps": index / max(elapsed, 0.001),
            "segments": len(store.segments()),
            "events": store.event_counts(),
        }
        (project / "analysis_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        with (project / "review_events.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "clip_seconds", "source_seconds", "reason"])
            for frame_index, kind in store.db.execute(
                "SELECT frame,kind FROM events ORDER BY frame"
            ):
                timestamp = frame_index / info["fps"]
                writer.writerow(
                    [
                        frame_index,
                        round(timestamp, 4),
                        round(timestamp + info["source_start_seconds"], 4),
                        kind,
                    ]
                )
        return summary
    finally:
        cap.release()
        store.close()


def reset_incomplete_analysis(project):
    """Discard only incomplete analysis metadata; never delete videos or models."""
    project = Path(project).resolve()
    database = project / "analysis.sqlite"
    if not database.exists():
        return
    store = Store(database)
    complete = store.get_meta("complete", False)
    store.close()
    if complete:
        raise ValueError("Completed analysis is preserved. Use another project for re-analysis.")
    for suffix in ("", "-wal", "-shm"):
        Path(str(database) + suffix).unlink(missing_ok=True)
