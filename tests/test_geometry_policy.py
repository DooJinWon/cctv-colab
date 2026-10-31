from copy import deepcopy
import numpy as np
import pytest
from cctv_privacy.geometry import clip_box, inverse_rotated_box, conservative_merge
from cctv_privacy.pipeline import AnalysisConfig, associate
from cctv_privacy.policy import EditPlan, mask_boxes, manual_box
from cctv_privacy.render import redact


def record():
    return {
        "index": 5,
        "people": [],
        "held_boxes": [],
        "flags": [],
        "faces": [
            {
                "box": [20, 20, 40, 40],
                "segment_id": "a",
                "reveal_eligible": True,
                "ambiguous": False,
            },
            {
                "box": [65, 20, 85, 40],
                "segment_id": "b",
                "reveal_eligible": True,
                "ambiguous": False,
            },
        ],
    }


def approved(**kwargs):
    return EditPlan(
        mode="selected",
        reveal_intervals=[
            {"segment_id": "a", "start_frame": 3, "end_frame": 8, "confirmed_same_person": True}
        ],
        **kwargs,
    )


@pytest.mark.parametrize(
    "k,rotated",
    [(0, [10, 20, 30, 40]), (1, [20, 70, 40, 90]), (2, [70, 40, 90, 60]), (3, [40, 10, 60, 30])],
)
def test_rotation_inverse_pixel_edges(k, rotated):
    assert inverse_rotated_box(rotated, k, 100, 80) == [10, 20, 30, 40]


@pytest.mark.parametrize(
    "box", [[1, 1, 1, 5], [5, 5, 2, 2], [float("nan"), 0, 1, 1], [-10, -5, -2, -1]]
)
def test_invalid_boxes_are_empty(box):
    assert clip_box(box, 100, 100) is None


def test_duplicate_merge_uses_union_not_smaller_box():
    merged = conservative_merge(
        [{"box": [10, 10, 30, 30], "score": 0.9}, {"box": [12, 8, 32, 28], "score": 0.8}]
    )
    assert len(merged) == 1
    assert merged[0]["box"] == [10, 8, 32, 30]


def test_all_mode_masks_even_approved_target():
    p = approved()
    p.mode = "all"
    assert len(mask_boxes(record(), p, 100, 100)) == 2


def test_reveal_only_inside_reviewed_interval():
    r = record()
    assert len(mask_boxes(r, approved(), 100, 100)) == 1
    r["index"] = 9
    assert len(mask_boxes(r, approved(), 100, 100)) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("reveal_eligible", False),
        ("ambiguous", True),
        ("segment_id", None),
        ("segment_id", "new-segment"),
    ],
)
def test_uncertain_or_reacquired_target_stays_masked(field, value):
    r = record()
    r["faces"][0][field] = value
    assert len(mask_boxes(r, approved(), 100, 100)) == 2


def test_untracked_face_stays_masked():
    r = record()
    r["faces"].append({"box": [1, 70, 10, 80]})
    assert len(mask_boxes(r, approved(), 100, 100)) == 2


def test_person_without_face_uses_whole_person_including_lying_pose():
    r = {"index": 5, "faces": [], "people": [{"box": [10, 50, 95, 75], "face_count": 0}]}
    boxes = mask_boxes(r, approved(), 100, 100)
    assert boxes[0][0] <= 10 and boxes[0][2] >= 95
    assert boxes[0][1] <= 50 and boxes[0][3] >= 75


def test_manual_and_other_person_masks_override_target():
    r = record()
    r["faces"][1]["box"] = [30, 25, 50, 45]
    p = approved(manual_masks=[{"keyframes": [{"frame": 5, "box": [0.2, 0.2, 0.3, 0.3]}]}])
    frame = np.full((100, 100, 3), 200, np.uint8)
    result = redact(frame, mask_boxes(r, p, 100, 100), "solid")
    assert np.all(result[25, 25] == 0)
    assert np.all(result[35, 35] == 0)


@pytest.mark.parametrize("style", ["solid", "mosaic", "blur"])
def test_redaction_preserves_outside_boxes(style):
    frame = np.random.default_rng(2).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    result = redact(frame, [[20, 20, 60, 60]], style)
    assert np.array_equal(result[:20], frame[:20])
    assert not np.array_equal(result[20:60, 20:60], frame[20:60, 20:60])


def test_manual_mask_moves_between_keyframes_and_has_finite_interval():
    mask = {
        "keyframes": [
            {"frame": 0, "box": [0.1, 0.2, 0.3, 0.4]},
            {"frame": 10, "box": [0.5, 0.2, 0.7, 0.4]},
        ]
    }
    assert manual_box(mask, 5, 100, 100) == [30, 20, 50, 40]
    assert manual_box(mask, 11, 100, 100) is None


def test_two_simultaneous_targets_are_rejected():
    p = approved()
    p.reveal_intervals.append(
        {"segment_id": "b", "start_frame": 5, "end_frame": 10, "confirmed_same_person": True}
    )
    with pytest.raises(ValueError, match="Two different"):
        p.validate()


def test_confirmation_is_required():
    p = approved()
    p.reveal_intervals[0]["confirmed_same_person"] = False
    with pytest.raises(ValueError, match="confirmation"):
        p.validate()


def test_ambiguous_spatial_association_never_reveals():
    people = [
        {"box": [0, 0, 90, 90], "segment_id": "a", "stable_frames": 5},
        {"box": [10, 0, 100, 90], "segment_id": "b", "stable_frames": 5},
    ]
    faces = [{"box": [20, 20, 40, 40]}]
    flags = associate(people, faces, AnalysisConfig())
    assert faces[0]["ambiguous"] and not faces[0]["reveal_eligible"]
    assert "ambiguous_face" in flags


def test_multiple_faces_inside_one_person_never_reveal():
    people = [{"box": [0, 0, 100, 100], "segment_id": "a", "stable_frames": 5}]
    faces = [{"box": [10, 10, 30, 30]}, {"box": [60, 10, 80, 30]}]
    associate(people, faces, AnalysisConfig())
    assert all(not f["reveal_eligible"] for f in faces)


def test_out_of_range_edit_rejected():
    with pytest.raises(ValueError, match="beyond"):
        approved().validate(frame_count=5)
