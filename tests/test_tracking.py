from types import SimpleNamespace
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from cctv_privacy.detectors import PersonDetectorTracker


class Model:
    def __init__(self):
        self.rows = []

    def predict(self, *args, **kwargs):
        data = np.asarray(self.rows, dtype=np.float32).reshape(-1, 6)
        return [SimpleNamespace(boxes=Boxes(data, orig_shape=(100, 200)))]


def make_tracker():
    detector = PersonDetectorTracker.__new__(PersonDetectorTracker)
    detector.device = "cpu"
    detector.image_size = 640
    detector.threshold = 0.1
    detector.model = Model()
    detector.tracker = BYTETracker(
        SimpleNamespace(
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        ),
        frame_rate=30,
    )
    detector.previous = {}
    detector.shot = 0
    detector.serial = 0
    return detector


def test_low_confidence_raw_detection_does_not_steal_high_confidence_track():
    detector = make_tracker()
    detector.model.rows = [[120, 10, 170, 90, 0.15, 0], [10, 10, 60, 90, 0.95, 0]]
    people = detector.detect(np.zeros((100, 200, 3), np.uint8), 0)
    assert people[0]["segment_id"] is None
    assert people[1]["segment_id"] is not None


def test_reappearance_after_gap_creates_new_reveal_segment():
    detector = make_tracker()
    frame = np.zeros((100, 200, 3), np.uint8)
    detector.model.rows = [[10, 10, 60, 90, 0.95, 0]]
    first = detector.detect(frame, 0)[0]
    detector.model.rows = []
    detector.detect(frame, 1)
    detector.model.rows = [[10, 10, 60, 90, 0.95, 0]]
    second = detector.detect(frame, 2)[0]
    assert first["segment_id"] != second["segment_id"]
    assert second["stable_frames"] == 1


def test_scene_reset_does_not_reuse_segment_identity():
    detector = make_tracker()
    frame = np.zeros((100, 200, 3), np.uint8)
    detector.model.rows = [[10, 10, 60, 90, 0.95, 0]]
    first = detector.detect(frame, 0)[0]
    detector.reset()
    second = detector.detect(frame, 1)[0]
    assert first["segment_id"] != second["segment_id"]
