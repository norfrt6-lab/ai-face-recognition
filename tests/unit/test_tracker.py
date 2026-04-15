"""Unit tests for core.tracker.iou_tracker.IoUTracker.

Tests cover:
  - Empty frame input
  - Single face tracked across multiple frames (stable ID)
  - Two non-overlapping faces maintaining distinct IDs
  - Track expiry after max_age+1 unmatched frames
  - IoU=0 (non-overlapping boxes) creating a new track
  - IoU above threshold matching existing track
  - _compute_iou helper via FaceBox.iou() (the internal implementation)
"""

from __future__ import annotations

from typing import List

import pytest

from core.detector.base_detector import FaceBox
from core.tracker.iou_tracker import IoUTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box(x1: int, y1: int, x2: int, y2: int, conf: float = 0.9, idx: int = 0) -> FaceBox:
    """Convenience constructor for FaceBox."""
    return FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, face_index=idx)


def _track_ids(faces: List[FaceBox]) -> List[int]:
    """Extract track_id values from a list of FaceBox objects."""
    return [f.track_id for f in faces]


# ---------------------------------------------------------------------------
# Test: empty input
# ---------------------------------------------------------------------------


class TestIoUTrackerEmptyInput:
    def test_update_with_no_faces_returns_empty_list(self):
        """update() with an empty list should return an empty result."""
        tracker = IoUTracker()
        result = tracker.update([])
        assert result == []

    def test_empty_frame_increments_age_then_prunes(self):
        """Tracks should age and eventually be pruned on repeated empty frames."""
        tracker = IoUTracker(iou_threshold=0.3, max_age=2)
        face = _box(0, 0, 100, 100)
        # Establish a track
        tracker.update([face])
        assert tracker.active_tracks == 1

        # Feed empty frames until the track should be pruned (max_age + 1 empties)
        for _ in range(3):  # max_age=2 → pruned after 3 empty updates
            tracker.update([])

        assert tracker.active_tracks == 0


# ---------------------------------------------------------------------------
# Test: single face tracked across frames
# ---------------------------------------------------------------------------


class TestIoUTrackerSingleFace:
    def test_single_face_gets_consistent_track_id_across_frames(self):
        """The same face appearing in 3 consecutive frames should keep the same ID."""
        tracker = IoUTracker(iou_threshold=0.3, max_age=5)
        face = _box(10, 10, 90, 90)

        ids = []
        for _ in range(3):
            result = tracker.update([face])
            assert len(result) == 1
            ids.append(result[0].track_id)

        assert ids[0] == ids[1] == ids[2], (
            f"Expected the same track ID across 3 frames, got {ids}"
        )

    def test_first_detection_assigns_new_track_id(self):
        """The first update should create a track with a positive integer ID."""
        tracker = IoUTracker()
        result = tracker.update([_box(0, 0, 50, 50)])
        assert len(result) == 1
        assert result[0].track_id is not None
        assert result[0].track_id >= 1


# ---------------------------------------------------------------------------
# Test: two non-overlapping faces
# ---------------------------------------------------------------------------


class TestIoUTrackerTwoFaces:
    def test_two_non_overlapping_faces_get_distinct_ids(self):
        """Two faces that never overlap should maintain distinct track IDs."""
        tracker = IoUTracker(iou_threshold=0.3, max_age=5)
        face_left = _box(0, 0, 50, 50, idx=0)
        face_right = _box(200, 200, 250, 250, idx=1)

        ids_left = []
        ids_right = []

        for _ in range(3):
            result = tracker.update([face_left, face_right])
            assert len(result) == 2
            # Sort by x1 to get consistent ordering
            sorted_result = sorted(result, key=lambda f: f.x1)
            ids_left.append(sorted_result[0].track_id)
            ids_right.append(sorted_result[1].track_id)

        # Each face should have a consistent ID
        assert len(set(ids_left)) == 1, f"Left face IDs changed: {ids_left}"
        assert len(set(ids_right)) == 1, f"Right face IDs changed: {ids_right}"
        # The two faces must have different IDs
        assert ids_left[0] != ids_right[0]

    def test_two_faces_count_as_two_active_tracks(self):
        tracker = IoUTracker()
        tracker.update([_box(0, 0, 50, 50), _box(200, 200, 250, 250)])
        assert tracker.active_tracks == 2


# ---------------------------------------------------------------------------
# Test: track expiry
# ---------------------------------------------------------------------------


class TestIoUTrackerExpiry:
    def test_track_expires_after_max_age_plus_one_unmatched_frames(self):
        """A track should be dropped after max_age+1 consecutive unmatched frames."""
        max_age = 3
        tracker = IoUTracker(iou_threshold=0.3, max_age=max_age)
        face = _box(0, 0, 100, 100)

        # Establish the track
        tracker.update([face])
        assert tracker.active_tracks == 1

        # Send empty frames — track ages but is not yet pruned during max_age frames
        for i in range(max_age):
            tracker.update([])
            assert tracker.active_tracks == 1, (
                f"Track should still exist after {i + 1} empty frame(s) (max_age={max_age})"
            )

        # One more empty frame → track.age > max_age → pruned
        tracker.update([])
        assert tracker.active_tracks == 0

    def test_reappearing_face_gets_new_track_id_after_expiry(self):
        """A face that reappears after its track expires should get a new track ID."""
        tracker = IoUTracker(iou_threshold=0.3, max_age=1)
        face = _box(0, 0, 100, 100)

        result1 = tracker.update([face])
        original_id = result1[0].track_id

        # Expire the track (max_age=1 → 2 empty frames)
        tracker.update([])
        tracker.update([])
        assert tracker.active_tracks == 0

        # Reappear
        result2 = tracker.update([face])
        new_id = result2[0].track_id

        assert new_id != original_id, (
            f"Expected a new track ID after expiry, but got {new_id} == {original_id}"
        )


# ---------------------------------------------------------------------------
# Test: IoU = 0 creates a new track
# ---------------------------------------------------------------------------


class TestIoUTrackerZeroIoU:
    def test_non_overlapping_box_creates_new_track(self):
        """A detection with IoU=0 against all existing tracks → new track."""
        tracker = IoUTracker(iou_threshold=0.3, max_age=5)
        face_a = _box(0, 0, 50, 50)
        face_b = _box(200, 200, 300, 300)  # no overlap with face_a

        result_a = tracker.update([face_a])
        id_a = result_a[0].track_id

        result_b = tracker.update([face_b])
        id_b = result_b[0].track_id

        # face_b has zero overlap with face_a's track → must be a new track
        assert id_b != id_a
        assert tracker.active_tracks == 2  # both tracks are alive (face_a aged once)

    def test_iou_zero_between_adjacent_non_overlapping_boxes(self):
        """FaceBox.iou() should return 0.0 for non-overlapping boxes."""
        box1 = _box(0, 0, 50, 50)
        box2 = _box(50, 50, 100, 100)  # touches corner but no area overlap
        assert box1.iou(box2) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test: IoU above threshold matches existing track
# ---------------------------------------------------------------------------


class TestIoUTrackerMatching:
    def test_overlapping_box_matches_existing_track(self):
        """A detection with IoU above threshold should match and reuse the track ID."""
        tracker = IoUTracker(iou_threshold=0.3, max_age=5)
        face1 = _box(0, 0, 100, 100)
        # Slightly shifted version of face1 → IoU still well above 0.3
        face2 = _box(5, 5, 105, 105)

        result1 = tracker.update([face1])
        tid1 = result1[0].track_id

        result2 = tracker.update([face2])
        tid2 = result2[0].track_id

        assert tid1 == tid2, (
            f"Expected track ID {tid1} to be reused for overlapping detection, got {tid2}"
        )

    def test_iou_above_threshold_exact_calculation(self):
        """Verify FaceBox.iou() result for a known overlapping pair."""
        # box1: 0-100 × 0-100 (area 10000)
        # box2: 50-150 × 50-150 (area 10000)
        # intersection: 50-100 × 50-100 = 50×50 = 2500
        # union: 10000 + 10000 - 2500 = 17500
        # IoU = 2500/17500 ≈ 0.1429
        box1 = _box(0, 0, 100, 100)
        box2 = _box(50, 50, 150, 150)
        expected_iou = 2500 / 17500
        assert box1.iou(box2) == pytest.approx(expected_iou, rel=1e-5)

    def test_identical_boxes_have_iou_one(self):
        """Identical boxes should have IoU = 1.0."""
        box = _box(10, 10, 90, 90)
        assert box.iou(box) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test: IoU helper / _compute_iou reconstruction
# ---------------------------------------------------------------------------


class TestIoUHelperDirect:
    def test_iou_via_facebox_method_matches_manual_formula(self):
        """Test _compute_iou logic as implemented in FaceBox.iou()."""
        # Two boxes that fully overlap (same box)
        b = _box(20, 20, 80, 80)
        assert b.iou(b) == pytest.approx(1.0)

    def test_iou_completely_contained_box(self):
        """Inner box completely inside outer box — IoU = inner_area / outer_area."""
        outer = _box(0, 0, 100, 100)  # area = 10000
        inner = _box(25, 25, 75, 75)  # area = 2500
        # intersection = inner area = 2500; union = 10000
        expected = 2500 / 10000
        assert outer.iou(inner) == pytest.approx(expected, rel=1e-5)

    def test_iou_partially_overlapping(self):
        """Partial overlap — verify the formula."""
        # box1: x 0-60, y 0-60 → area 3600
        # box2: x 40-100, y 40-100 → area 3600
        # intersection: x 40-60, y 40-60 → 20×20 = 400
        # union: 3600+3600-400 = 6800
        box1 = _box(0, 0, 60, 60)
        box2 = _box(40, 40, 100, 100)
        expected = 400 / 6800
        assert box1.iou(box2) == pytest.approx(expected, rel=1e-5)

    def test_iou_zero_area_box(self):
        """A zero-area box (degenerate) should return IoU=0."""
        zero = _box(50, 50, 50, 50)  # zero width and height
        other = _box(0, 0, 100, 100)
        assert zero.iou(other) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: tracker reset
# ---------------------------------------------------------------------------


class TestIoUTrackerReset:
    def test_reset_clears_all_tracks_and_resets_id_counter(self):
        tracker = IoUTracker()
        tracker.update([_box(0, 0, 50, 50)])
        assert tracker.active_tracks == 1

        tracker.reset()
        assert tracker.active_tracks == 0

        # After reset the next track should start from ID 1 again
        result = tracker.update([_box(0, 0, 50, 50)])
        assert result[0].track_id == 1
