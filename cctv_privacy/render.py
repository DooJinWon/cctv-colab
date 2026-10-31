"""Rendering is independent of model inference and never silently skips frames."""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
from .geometry import clip_box
from .policy import EditPlan, mask_boxes
from .store import Store
from .video import Encoder, file_digest


def redact(frame, boxes, style="mosaic", block_pixels=18):
    result = frame.copy()
    height, width = frame.shape[:2]
    for box in boxes:
        clipped = clip_box(box, width, height)
        if not clipped:
            continue
        x1, y1, x2, y2 = clipped
        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        if style == "solid":
            replacement = np.zeros_like(roi)
        elif style == "mosaic":
            small = cv2.resize(
                roi,
                (max(1, w // block_pixels), max(1, h // block_pixels)),
                interpolation=cv2.INTER_AREA,
            )
            replacement = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif style == "blur":
            replacement = cv2.GaussianBlur(roi, (0, 0), sigmaX=max(15, min(w, h) / 3))
        else:
            raise ValueError(f"Unknown style: {style}")
        result[y1:y2, x1:x2] = replacement
    return result


def validate_project_plan(store, plan):
    if not store.get_meta("complete"):
        raise RuntimeError("Analysis is incomplete. Export is disabled.")
    info = store.get_meta("manifest")
    analysis_id = store.get_meta("analysis_id")
    if plan.analysis_id and plan.analysis_id != analysis_id:
        raise ValueError("This edit plan belongs to another analysis project.")
    if (plan.reveal_intervals or plan.manual_masks) and plan.analysis_id != analysis_id:
        raise ValueError("An edited plan must be bound to this analysis project.")
    plan.validate(info["frame_count"])
    segments = {s["id"]: s for s in store.segments()}
    for interval in plan.reveal_intervals:
        segment = segments.get(interval["segment_id"])
        if (
            not segment
            or interval["start_frame"] < segment["first_frame"]
            or interval["end_frame"] > segment["last_frame"]
        ):
            raise ValueError("Reveal interval must stay within its detected continuous segment.")
    return info


def render(project, output, plan=None, progress=None):
    project = Path(project).resolve()
    output = Path(output).resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output filename must end with .mp4.")
    if output == project / "prepared.mp4":
        raise ValueError("Cannot overwrite the prepared source video.")
    plan = plan or EditPlan()
    store = Store(project / "analysis.sqlite")
    encoder = None
    cap = None
    try:
        info = validate_project_plan(store, plan)
        if file_digest(project / info["prepared_name"]) != store.get_meta("prepared_sha256"):
            raise RuntimeError("Prepared video changed after analysis. Export is disabled.")
        cap = cv2.VideoCapture(str(project / info["prepared_name"]))
        if not cap.isOpened():
            raise RuntimeError("Prepared video could not be opened.")
        encoder = Encoder(output, info, project / info["prepared_name"], plan.keep_audio)
        started = time.perf_counter()
        for index in range(info["frame_count"]):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {index}. Export was aborted.")
            record = store.get(index)
            boxes = mask_boxes(record, plan, info["width"], info["height"])
            encoder.write(redact(frame, boxes, plan.style, plan.mosaic_block_pixels))
            if progress and (index % 30 == 0 or index + 1 == info["frame_count"]):
                progress(index + 1, info["frame_count"])
        ok, _ = cap.read()
        if ok:
            raise RuntimeError("Prepared video changed after analysis. Export was aborted.")
        output_info = encoder.finish(info["frame_count"])
        elapsed = time.perf_counter() - started
        digest = hashlib.sha256(json.dumps(asdict(plan), sort_keys=True).encode()).hexdigest()
        report = {
            "analysis_id": store.get_meta("analysis_id"),
            "output_name": output.name,
            "frames": info["frame_count"],
            "render_seconds": elapsed,
            "output_info": output_info,
            "edit_plan_sha256": digest,
            "edit_plan": asdict(plan),
            "automatic_review_flags": store.event_counts(),
            "review_status": "UNREVIEWED_EXPORT",
            "limitations": "Automated detection and tracking can miss or misassociate people. Manual review is required.",
        }
        output.with_suffix(".report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return report
    except BaseException:
        if encoder:
            encoder.abort()
        raise
    finally:
        if cap:
            cap.release()
        store.close()
