from app.services.tasks.base import collect_media, split_value_unit


def test_collect_media_splits_videos_and_images():
    clips = [
        type("C", (), {"converted_path": "/a/b.mp4", "is_video": True})(),
        type("C", (), {"converted_path": "/a/c.png", "is_video": False})(),
        type("C", (), {"converted_path": None, "is_video": False})(),
    ]
    imgs, vids = collect_media(clips)
    assert imgs == ["/a/c.png"]
    assert vids == ["/a/b.mp4"]


def test_split_value_unit_simple_number():
    v, u = split_value_unit("55 %")
    assert v == "55"
    assert u == "%"


def test_split_value_unit_plain_number():
    v, u = split_value_unit("1.23")
    assert v == "1.23"
    assert u is None


def test_split_value_unit_with_word_unit():
    v, u = split_value_unit("12 cm/s")
    assert v == "12"
    assert u == "cm/s"


def test_split_value_unit_none_when_non_numeric():
    v, u = split_value_unit("Unable to measure")
    assert v is None
    assert u is None
