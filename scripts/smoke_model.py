"""Optional network/model smoke test on a caller-provided, non-sensitive image."""

from pathlib import Path
import argparse
import json
import sys
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cctv_privacy.video import prepare_video
from cctv_privacy.pipeline import AnalysisConfig, analyze
from cctv_privacy.store import Store
from cctv_privacy.policy import EditPlan
from cctv_privacy.render import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("output_directory")
    args = parser.parse_args()
    root = Path(args.output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError("Image could not be read.")
    h, w = image.shape[:2]
    writer = cv2.VideoWriter(str(root / "input.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
    for _ in range(8):
        writer.write(image)
    writer.release()
    project = root / "project"
    prepare_video(root / "input.mp4", project, end_seconds=None)
    summary = analyze(
        project,
        AnalysisConfig(device="cpu", person_image_size=640, face_rotations=(0,), face_tile_size=0),
    )
    store = Store(project / "analysis.sqlite")
    sample = store.get(5)
    assert sample["people"], "No people detected in the fixture."
    assert sample["faces"], "No faces detected in the fixture."
    eligible = next((f for f in sample["faces"] if f["reveal_eligible"]), None)
    render(project, root / "all.mp4", EditPlan(style="solid"))
    if eligible:
        segment = next(s for s in store.segments() if s["id"] == eligible["segment_id"])
        plan = EditPlan(
            mode="selected",
            analysis_id=store.get_meta("analysis_id"),
            reveal_intervals=[
                {
                    "segment_id": segment["id"],
                    "start_frame": segment["first_frame"],
                    "end_frame": segment["last_frame"],
                    "confirmed_same_person": True,
                }
            ],
        )
        render(project, root / "selected.mp4", plan)
    store.close()
    from cctv_privacy.detectors import FaceDetector

    detector = FaceDetector(
        project / "models/face_detection_yunet_2023mar.onnx",
        rotations=(0, 90, 180, 270),
        tile_size=640,
    )
    augmented_faces = detector.detect(image)
    assert augmented_faces
    result = {
        "summary": summary,
        "people_in_frame": len(sample["people"]),
        "faces_in_frame": len(sample["faces"]),
        "augmented_faces_in_image": len(augmented_faces),
        "selected_export_tested": eligible is not None,
    }
    (root / "smoke_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
