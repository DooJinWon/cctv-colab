import json
import shutil
import subprocess
from pathlib import Path
import cv2
import pytest
from cctv_privacy.pipeline import AnalysisConfig, analyze, reset_incomplete_analysis
from cctv_privacy.policy import EditPlan
from cctv_privacy.render import render
from cctv_privacy.store import Store
from cctv_privacy.video import prepare_video, probe


class People:
    def reset(self):
        pass

    def detect(self, frame, index):
        return [
            {
                "box": [10, 5, 60, 90],
                "score": 0.95,
                "track_id": "0:1",
                "segment_id": "segment-000001",
                "stable_frames": index + 1,
            }
        ]


class Faces:
    def detect(self, frame):
        return [{"box": [25, 10, 45, 30], "score": 0.95}]


@pytest.fixture
def project(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for integration tests.")
    source = tmp_path / "source with spaces.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x96:rate=6:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    folder = tmp_path / "project with spaces"
    info = prepare_video(source, folder, end_seconds=None)
    assert info["frame_count"] == 12
    analyze(
        folder,
        AnalysisConfig(face_rotations=(0,), face_tile_size=0),
        detector_pair=(People(), Faces()),
    )
    return folder


@pytest.mark.parametrize("keep_audio", [False, True])
def test_prepare_analyze_render_preserves_every_frame_and_audio_policy(project, keep_audio):
    output = project / f"export-{keep_audio}.mp4"
    report = render(project, output, EditPlan(keep_audio=keep_audio))
    info = probe(output, count_frames=True)
    assert info["frame_count"] == 12
    assert info["has_audio"] == keep_audio
    assert abs(info["duration"] - 2) < 0.2
    assert report["review_status"] == "UNREVIEWED_EXPORT"
    assert output.with_suffix(".report.json").exists()


def test_selected_export_with_bound_plan(project):
    store = Store(project / "analysis.sqlite")
    plan = EditPlan(
        mode="selected",
        analysis_id=store.get_meta("analysis_id"),
        reveal_intervals=[
            {
                "segment_id": "segment-000001",
                "start_frame": 0,
                "end_frame": 11,
                "confirmed_same_person": True,
            }
        ],
    )
    store.close()
    render(project, project / "selected.mp4", plan)
    assert probe(project / "selected.mp4", count_frames=True)["frame_count"] == 12


def test_edits_cannot_cross_projects(project):
    with pytest.raises(ValueError, match="another analysis"):
        render(project, project / "invalid.mp4", EditPlan(analysis_id="wrong-project"))


def test_changed_source_blocks_export(project):
    with (project / "prepared.mp4").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(RuntimeError, match="changed after analysis"):
        render(project, project / "invalid.mp4")
    assert not (project / "invalid.mp4").exists()


def test_incomplete_analysis_blocks_export_and_can_be_reset(project):
    store = Store(project / "analysis.sqlite")
    store.set_meta("complete", False)
    store.close()
    with pytest.raises(RuntimeError, match="incomplete"):
        render(project, project / "invalid.mp4")
    reset_incomplete_analysis(project)
    assert not (project / "analysis.sqlite").exists()
    assert (project / "prepared.mp4").exists()


def test_completed_analysis_is_not_reset(project):
    with pytest.raises(ValueError, match="Completed"):
        reset_incomplete_analysis(project)


def test_missing_analysis_frame_aborts_without_publishing_partial_output(project):
    store = Store(project / "analysis.sqlite")
    store.db.execute("DELETE FROM frames WHERE idx=5")
    store.close()
    with pytest.raises(ValueError, match="프레임"):
        render(project, project / "invalid.mp4")
    assert not (project / "invalid.mp4").exists()
    assert not (project / "invalid.partial.mp4").exists()


def test_existing_output_is_not_overwritten(project):
    output = project / "existing.mp4"
    output.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        render(project, output)
    assert output.read_bytes() == b"preserve me"


def test_review_widgets_save_edits_and_manual_masks(project):
    from cctv_privacy.review import ReviewUI

    ui = ReviewUI(project)
    try:
        ui.target.value = "segment-000001"
        ui.confirm.value = True
        ui._add_reveal(None)
        assert len(ui.plan.reveal_intervals) == 1
        ui.mode.value = "selected"
        ui._capture_box("start")
        ui._add_mask(None)
        assert len(ui.plan.manual_masks) == 1
        saved = EditPlan.load(project / "edits.json")
        assert saved.mode == "selected"
        assert saved.manual_masks
        ui.frame.value = 5
        assert ui.image.value
    finally:
        ui.close()


def test_cfr_normalization_and_clip_bounds(tmp_path):
    source = tmp_path / "variable.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=128x96:rate=12:duration=3",
            "-vf",
            "select='not(mod(n,3))'",
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    info = prepare_video(
        source, tmp_path / "normalized", start_seconds=0.5, end_seconds=2.0, fps=10
    )
    assert info["frame_count"] == 15
    assert info["width"] == 128 and info["height"] == 96
    assert not info["has_audio"]
