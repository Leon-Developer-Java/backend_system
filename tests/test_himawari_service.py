import json

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
