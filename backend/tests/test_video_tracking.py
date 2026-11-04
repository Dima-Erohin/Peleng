import cv2
import numpy as np
import pytest

from peleng.video_tracking import TemplateTracker
from scripts.track_video import main, track_video


def fixture():
    rng = np.random.default_rng(42)
    target = rng.integers(10, 240, (24, 32), dtype=np.uint8)
    image = np.full((180, 240), 110, np.uint8)
    image[55:79, 70:102] = target
    return image, target


def test_tracks_real_pixels_with_translation_and_scale():
    first, target = fixture()
    tracker = TemplateTracker(first, (70, 55, 32, 24))
    for x, y in [(73, 56), (80, 60), (90, 65)]:
        image = np.full_like(first, 110)
        image[y:y + 24, x:x + 32] = target
        match = tracker.update(image)
        assert match.bbox == (x, y, 32, 24)
        assert match.correlation > 0.99
    image = np.full_like(first, 110)
    resized = cv2.resize(target, (round(32 * 1.15), round(24 * 1.15)))
    image[67:67 + resized.shape[0], 95:95 + resized.shape[1]] = resized
    assert tracker.update(image).bbox == (95, 67, resized.shape[1], resized.shape[0])


def test_lost_has_no_invented_box_and_can_reacquire_short_gap():
    first, _ = fixture()
    tracker = TemplateTracker(first, (70, 55, 32, 24))
    assert tracker.update(np.full_like(first, 110)).bbox is None
    assert tracker.update(first).bbox == (70, 55, 32, 24)


def test_long_loss_requires_reinitialization_and_ambiguous_peaks_rejected():
    first, target = fixture()
    tracker = TemplateTracker(first, (70, 55, 32, 24), max_lost=1)
    for _ in range(2):
        assert tracker.update(np.full_like(first, 110)).bbox is None
    assert tracker.update(first).bbox is None
    ambiguous = first.copy()
    ambiguous[55:79, 120:152] = target
    assert TemplateTracker(first, (70, 55, 32, 24)).update(ambiguous).bbox is None


def test_rejects_invalid_seed_threshold_and_dimensions():
    first, _ = fixture()
    for roi in [(-1, 0, 32, 24), (230, 0, 32, 24), (0, 0, 2, 2), (0, 0, 32, 24)]:
        with pytest.raises(ValueError):
            TemplateTracker(first, roi)
    with pytest.raises(ValueError):
        TemplateTracker(first, (70, 55, 32, 24), threshold=float("nan"))
    with pytest.raises(ValueError):
        TemplateTracker(first, (70, 55, 32, 24)).update(np.zeros((10, 10), np.uint8))


def test_video_export_decodes_both_directions_and_preserves_original(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    path = tmp_path / "target.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (240, 180))
    assert writer.isOpened()
    _, target = fixture()
    for i in range(12):
        image = np.full((180, 240), 110, np.uint8)
        image[55:79, 70 + i:102 + i] = target
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = track_video(path, 5, (75, 55, 32, 24), "test-cam", "drone-test")
    assert result["source_sha256"] == digest
    assert result["space"] == "image"
    assert result["frame_count"] == 12
    assert result["visible_frames"] == 12
    assert result["frames"][5]["status"] == "seed"
    for i, row in enumerate(result["frames"]):
        assert row["time"] == i / 30
        assert abs(row["bbox"][0] - (70 + i)) <= 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError):
        track_video(path, 15, (75, 55, 32, 24), "test", "test")
    with pytest.raises(ValueError):
        track_video(tmp_path / "missing.mp4", 0, (0, 0, 32, 24), "test", "test")
    # A source named like the old temporary output must never be overwritten.
    source = path.rename(tmp_path / "track.json.tmp")
    output = tmp_path / "track.json"
    args = [
        "track_video", str(source), "--seed-frame", "5",
        "--roi", "75", "55", "32", "24", "--output", str(output),
    ]
    monkeypatch.setattr(sys, "argv", args)
    main()
    assert json.loads(output.read_text())["visible_frames"] == 12
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert set(tmp_path.iterdir()) == {source, output}
    monkeypatch.setattr(sys, "argv", [*args[:-1], str(source)])
    with pytest.raises(SystemExit):
        main()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
