import json
import os

from services import himawari_service


def test_display_data_reads_nested_scene_meta_and_png(tmp_path, monkeypatch):
    scene_dir = tmp_path / "Himawari" / "20260616" / "0000"
    (scene_dir / "meta").mkdir(parents=True)
    (scene_dir / "latlon").mkdir()
    (scene_dir / "composites").mkdir()
    png = scene_dir / "latlon" / "B13.png"
    png.write_bytes(b"png")
    composite = scene_dir / "composites" / "true_color.png"
    composite.write_bytes(b"png")
    (scene_dir / "meta" / "scene.meta.json").write_text(
        json.dumps(
            {
                "scene_id": "20260616_0000",
                "grid": {"nx": 1576, "ny": 901},
                "extent": [73, 18, 136, 54],
                "variables": [{"key": "B13", "png": png.as_posix()}],
                "composites": [{"key": "true_color", "png": composite.as_posix()}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(himawari_service, "DATA_DIR", tmp_path / "Himawari")

    payload = himawari_service.get_display_data()

    assert payload["meta_json"]["scene_id"] == "20260616_0000"
    assert payload["png"] == composite.as_posix()
    assert payload["png_data_url"].startswith("data:image/png;base64,")
    assert payload["extent"] == [73, 18, 136, 54]
    assert payload["grid"] == {"nx": 1576, "ny": 901}
    assert payload["variables"][0]["key"] == "B13"
    assert payload["variables"][0]["png_data_url"].startswith("data:image/png;base64,")
    assert payload["composites"][0]["key"] == "true_color"
    assert payload["composites"][0]["png_data_url"].startswith("data:image/png;base64,")
    assert payload["png_files"] == [png.as_posix()]


def test_display_data_resolves_committed_pngs_when_meta_has_other_machine_absolute_paths(tmp_path, monkeypatch):
    scene_dir = tmp_path / "Himawari" / "20260616" / "0000"
    (scene_dir / "meta").mkdir(parents=True)
    (scene_dir / "latlon").mkdir()
    (scene_dir / "composites").mkdir()
    png = scene_dir / "latlon" / "B13.png"
    png.write_bytes(b"png")
    composite = scene_dir / "composites" / "true_color.png"
    composite.write_bytes(b"png")
    (scene_dir / "meta" / "scene.meta.json").write_text(
        json.dumps(
            {
                "scene_id": "20260616_0000",
                "grid": {"nx": 1576, "ny": 901},
                "extent": [73, 18, 136, 54],
                "variables": [{"key": "B13", "png": "/other/machine/data/Himawari/20260616/0000/latlon/B13.png"}],
                "composites": [{"key": "true_color", "png": "/other/machine/data/Himawari/20260616/0000/composites/true_color.png"}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(himawari_service, "DATA_DIR", tmp_path / "Himawari")

    payload = himawari_service.get_display_data()

    assert payload["png"] == composite.as_posix()
    assert payload["png_data_url"].startswith("data:image/png;base64,")
    assert payload["variables"][0]["png_data_url"].startswith("data:image/png;base64,")
    assert payload["composites"][0]["png_data_url"].startswith("data:image/png;base64,")


def test_display_data_prefers_later_scene_when_meta_timestamps_match(tmp_path, monkeypatch):
    for time in ("0000", "0020"):
        scene_dir = tmp_path / "Himawari" / "20260616" / time
        (scene_dir / "meta").mkdir(parents=True)
        (scene_dir / "latlon").mkdir()
        png = scene_dir / "latlon" / "B13.png"
        png.write_bytes(b"png")
        meta = scene_dir / "meta" / "scene.meta.json"
        meta.write_text(
            json.dumps(
                {
                    "scene_id": f"20260616_{time}",
                    "grid": {"nx": 1576, "ny": 901},
                    "extent": [73, 18, 136, 54],
                    "variables": [{"key": "B13", "png": png.as_posix()}],
                    "composites": [],
                }
            ),
            encoding="utf-8",
        )
        os.utime(meta, (1_000_000, 1_000_000))

    monkeypatch.setattr(himawari_service, "DATA_DIR", tmp_path / "Himawari")

    payload = himawari_service.get_display_data()

    assert payload["meta_json"]["scene_id"] == "20260616_0020"
