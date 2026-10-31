# Validation record

## Executed checks

- Local environment: Apple Silicon Mac, Python 3.9.6, PyTorch 2.8.0, OpenCV 4.11.0, Ultralytics 8.3.241, system FFmpeg/ffprobe.
- Final automated suite: **46 tests passed** locally (41 processing/tracking/widget tests and 5 notebook packaging tests).
- Notebook packaging checks validate the notebook schema, compile every code cell, confirm outputs are cleared, compare every embedded source file with the repository, and execute source extraction with dependency installation disabled.
- All **10 executable notebook cells** were also run in order in a fresh local CPU process, substituting a public test video and disabling package reinstallation/browser download APIs. The run extracted the embedded source, analyzed eight frames with the actual models, applied a reviewed target interval, rendered the result, and verified that the result ZIP excludes the unredacted prepared video.
- Synthetic video integration tests exercise actual FFmpeg decoding/encoding, clipping, CFR normalization, exact saved frame counts, and both audio policies.
- Synthetic detections exercise analysis, SQLite persistence, reveal plans, manual masks, edit-plan binding, source fingerprint rejection, partial-export cleanup, and notebook widget callbacks.
- ByteTrack tests use the actual installed tracker with synthetic detections, including a regression test for confidence-filtered detection indices and reappearance segmentation.

## Real-model smoke test

Executed successfully on the public Ultralytics `bus.jpg` fixture, converted into an eight-frame recorded video. The fixture and downloaded weights remain outside the repository.

- Actual YOLO11n and YuNet weights were loaded, not replaced by mock detectors.
- Four person boxes and two face boxes were present in the inspected prepared frame.
- Rotated/tiled YuNet inference on the source image returned three face regions.
- Both mask-all and selected-target exports completed and passed saved frame-count validation.
- A decoded mask-all output frame was visually inspected; face masks and whole-person fallback masks were present.
- The smoke clip is repeated imagery, not an accuracy benchmark or proof of identity continuity.
- A transient model-host connection failure was reproduced during development. The downloader now tries the official GitHub binary route and the OpenCV-maintained Hugging Face mirror; it rejects short Git LFS pointer responses.

## Not yet established

- End-to-end execution in the user's Google Colab or UIUC runtime.
- NVIDIA CUDA behavior and GPU memory/throughput measurements.
- Accuracy on the user's CCTV, including sleeping/lying children, low light, long occlusions, dense crowds, and camera-specific compression.
- Identity continuity or zero missed-face exposure across a full recording.
- Visual review by the user and suitability for releasing any real recording.

The project contains no private CCTV fixtures, credentials, or embedded model weights. An exported report remains `UNREVIEWED_EXPORT` even after the automated tests pass.
