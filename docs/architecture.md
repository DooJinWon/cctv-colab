# Architecture and data contracts

## Two-pass processing

`prepare_video` creates a CFR, auto-oriented working clip. `analyze` runs detection/tracking and writes each normalized frame to SQLite. `render` uses this cached analysis and an edit plan; model inference is not repeated. A project can have multiple exports with different edit plans.

## Coordinates and time

- Pixel boxes use half-open `(x1, y1, x2, y2)` coordinates in the prepared video's dimensions.
- Rotation augmentation maps all four box corners back to prepared-video coordinates.
- Duplicate face boxes are merged by union rather than shrinking the covered region.
- Manual-mask boxes use normalized coordinates in `[0, 1]` and are converted at render time.
- Reveal and manual-mask time ranges use inclusive prepared frame indices.
- Clip timestamp is `frame_index / prepared_fps`; source timestamp adds `source_start_seconds`.
- The manifest records the exact rational prepared FPS. Original VFR frame indices are not preserved.

## Frame schema

```json
{
  "index": 42,
  "timestamp": 1.4,
  "people": [{
    "box": [100, 50, 200, 350],
    "score": 0.93,
    "track_id": "0:3",
    "segment_id": "segment-000004",
    "stable_frames": 12,
    "face_count": 1,
    "crowded": false
  }],
  "faces": [{
    "box": [120, 70, 160, 110],
    "score": 0.87,
    "person_index": 0,
    "segment_id": "segment-000004",
    "ambiguous": false,
    "reveal_eligible": true
  }],
  "held_boxes": [],
  "flags": []
}
```

Detector confidence estimates detection quality; it is not the probability of biometric identity. `segment_id` is an application-generated ID for a continuous visible track fragment. A missing frame, candidate shot change, or large displacement can end that fragment.

## Reveal decision

```text
reveal(face, frame) =
    selected-target mode
    AND exactly one approved segment at this frame
    AND the face belongs to that segment
    AND the approved interval includes this frame
    AND the interval carries manual same-person confirmation
    AND the face is unambiguously linked to one detected person
    AND that person has exactly one associated face
    AND tracking has been stable for the configured minimum
    AND no detected person-box overlap exceeds the crowding threshold
```

Every other detected face is masked. People without an associated face are masked over their full body box. Held face regions and all manual masks are added afterward. The implementation never pastes an original target patch over existing masks.

## Failure behavior

- Inference/decoding exceptions leave analysis incomplete and disable rendering.
- The prepared video SHA-256 is recorded during analysis and checked before export.
- Edit plans with reveal/manual edits must reference the current analysis UUID.
- Export checks that every metadata row and every prepared frame exists.
- FFmpeg writes to a partial file. The final filename is published only after encoding succeeds and the saved frame count matches.
- Export report status remains `UNREVIEWED_EXPORT`; creating a file is not a claim of complete de-identification.

## Memory and lifecycle

Decoded frames, inference tensors, recent tracking state, and short-lived held regions are transient. Per-frame metadata is stored in SQLite. The notebook caches only the currently displayed preview frame. Models are cached on disk and are not bundled into Git or the notebook payload.

Interrupted analysis cannot safely resume from a frame number because tracker state is temporal. `reset_incomplete_analysis` discards only incomplete SQLite metadata and retains the prepared video/models. Completed analysis is never discarded by that helper.
