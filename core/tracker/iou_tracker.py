"""Simple IoU-based face tracker for temporal consistency across video frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.detector.base_detector import FaceBox


@dataclass
class _Track:
    track_id: int
    last_box: FaceBox
    age: int = 0  # frames since last matched
    total_frames: int = 1  # total frames this track has been alive


class IoUTracker:
    """
    Match detected faces across frames using IoU overlap to assign
    consistent track IDs.

    Args:
        iou_threshold: Minimum IoU to consider a detection-to-track match.
        max_age:       Drop a track after this many consecutive unmatched frames.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: Dict[int, _Track] = {}
        self._next_id: int = 1

    def update(self, faces: List[FaceBox]) -> List[Optional[FaceBox]]:
        """
        Match *faces* against existing tracks and return a new list with
        ``track_id`` populated on each face.

        Unmatched detections get new track IDs. Tracks that go unmatched
        for ``max_age`` consecutive frames are pruned.
        """
        if not faces:
            self._age_and_prune()
            return []

        if not self._tracks:
            return self._init_tracks(faces)

        # Build IoU cost matrix: tracks × detections
        track_ids = list(self._tracks.keys())
        iou_matrix: List[List[float]] = []
        for tid in track_ids:
            row = [self._tracks[tid].last_box.iou(f) for f in faces]
            iou_matrix.append(row)

        # Greedy matching: pick highest IoU pair iteratively
        matched_tracks: set = set()
        matched_dets: set = set()
        pairs: List[Tuple[int, int]] = []  # (track_idx, det_idx)

        # Flatten and sort all (track_idx, det_idx, iou) by iou descending
        candidates = []
        for ti, row in enumerate(iou_matrix):
            for di, score in enumerate(row):
                if score >= self.iou_threshold:
                    candidates.append((score, ti, di))
        candidates.sort(reverse=True)

        for _score, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            pairs.append((ti, di))
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Update matched tracks
        result: List[Optional[FaceBox]] = [None] * len(faces)
        for ti, di in pairs:
            tid = track_ids[ti]
            track = self._tracks[tid]
            face = faces[di]
            track.last_box = face
            track.age = 0
            track.total_frames += 1
            result[di] = self._with_track_id(face, tid)

        # Create new tracks for unmatched detections
        for di in range(len(faces)):
            if di not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(track_id=tid, last_box=faces[di])
                result[di] = self._with_track_id(faces[di], tid)

        # Age unmatched tracks and prune stale ones
        for ti, tid in enumerate(track_ids):
            if ti not in matched_tracks:
                self._tracks[tid].age += 1
        self._prune()

        return [r for r in result if r is not None]

    def reset(self) -> None:
        """Clear all tracks and reset ID counter."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def _init_tracks(self, faces: List[FaceBox]) -> List[FaceBox]:
        result = []
        for face in faces:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(track_id=tid, last_box=face)
            result.append(self._with_track_id(face, tid))
        return result

    def _age_and_prune(self) -> None:
        for track in self._tracks.values():
            track.age += 1
        self._prune()

    def _prune(self) -> None:
        stale = [tid for tid, t in self._tracks.items() if t.age > self.max_age]
        for tid in stale:
            del self._tracks[tid]

    @staticmethod
    def _with_track_id(face: FaceBox, track_id: int) -> FaceBox:
        return FaceBox(
            x1=face.x1,
            y1=face.y1,
            x2=face.x2,
            y2=face.y2,
            confidence=face.confidence,
            face_index=face.face_index,
            landmarks=face.landmarks,
            track_id=track_id,
        )
