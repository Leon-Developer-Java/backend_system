import numpy as np
import xarray as xr
from PIL import Image

from adapters import himawari_adapter


def test_all_ahi_bands_have_plain_language_metadata():
    expected = {f"B{i:02d}" for i in range(1, 17)}

    assert set(himawari_adapter.BAND_CATALOG) == expected

    for band, meta in himawari_adapter.BAND_CATALOG.items():
        assert meta["key"] == band
        assert meta["name_zh"]
        assert meta["plain_name"]
        assert meta["description"]
        assert meta["wavelength"]
        assert meta["unit"] in {"%", "K"}
        assert meta["display_unit"]
        assert meta["category"]
        assert meta["uses"]
        assert meta["cautions"]


def test_latlon_grid_uses_project_extent_and_resolution():
    grid = himawari_adapter.build_latlon_grid()

    assert grid["projection"] == "EPSG:4326"
    assert grid["grid_type"] == "regular_latlon"
    assert grid["extent"] == [73, 18, 136, 54]
    assert grid["resolution"] == 0.04
    assert grid["nx"] == 1576
    assert grid["ny"] == 901


def test_write_latlon_variable_outputs_png_float32_and_netcdf(tmp_path):
    data = np.array([[250.0, 260.0], [270.0, np.nan]], dtype=np.float32)
    grid = {
        "projection": "EPSG:4326",
        "grid_type": "regular_latlon",
        "extent": [73, 18, 73.04, 18.04],
        "resolution": 0.04,
        "nx": 2,
        "ny": 2,
    }

    result = himawari_adapter.write_latlon_variable(tmp_path, "B13", data, grid)

    assert result["key"] == "B13"
    assert result["grid"] == {"nx": 2, "ny": 2}
    assert result["extent"] == [73, 18, 73.04, 18.04]
    assert result["stats"]["min"] == 250.0
    assert result["stats"]["max"] == 270.0
    assert result["stats"]["mean"] == 260.0

    png = tmp_path / "latlon" / "B13.png"
    raw = tmp_path / "latlon" / "B13.float32"
    nc = tmp_path / "latlon" / "B13.nc"
    assert png.exists()
    assert raw.exists()
    assert nc.exists()
    assert np.fromfile(raw, dtype=np.float32).shape == (4,)
    assert Image.open(png).size == (2, 2)
    dataset = xr.open_dataset(nc)
    try:
        assert dataset["B13"].shape == (2, 2)
    finally:
        dataset.close()


def test_write_scene_metadata_indexes_variables_and_composites(tmp_path):
    grid = himawari_adapter.build_latlon_grid()
    variable = {"key": "B13", "name_zh": "红外窗口亮温", "png": "latlon/B13.png", "float32": "latlon/B13.float32", "netcdf": "latlon/B13.nc", "stats": {"min": 250.0, "max": 270.0, "mean": 260.0}}
    composite = {"key": "true_color", "name_zh": "真彩色云图", "png": "composites/true_color.png"}

    meta = himawari_adapter.write_scene_metadata(tmp_path, "20260616", "0000", "Himawari-9", tmp_path / "raw", 90, grid, [variable], [composite])

    assert (tmp_path / "meta" / "scene.meta.json").exists()
    assert meta["scene_id"] == "20260616_0000"
    assert meta["projection"] == "EPSG:4326"
    assert meta["grid"] == {"nx": 1576, "ny": 901}
    assert meta["variables"] == [variable]
    assert meta["composites"] == [composite]


def test_process_scene_reuses_existing_products_without_satpy(tmp_path):
    raw = tmp_path / "input" / "raw"
    raw.mkdir(parents=True)
    (raw / "HS_H09_20260616_0000_B13_FLDK_R20_S0110.DAT.bz2").write_bytes(b"")
    scene_dir = tmp_path / "out" / "20260616" / "0000"
    (scene_dir / "meta").mkdir(parents=True)
    (scene_dir / "latlon").mkdir()
    png = scene_dir / "latlon" / "B13.png"
    png.write_bytes(b"png")
    expected = himawari_adapter.write_scene_metadata(
        scene_dir,
        "20260616",
        "0000",
        "Himawari-9",
        raw,
        1,
        himawari_adapter.build_latlon_grid(),
        [{"key": "B13", "png": png.as_posix()}],
        [],
    )

    assert himawari_adapter.process_scene(raw, output_root=tmp_path / "out") == expected
