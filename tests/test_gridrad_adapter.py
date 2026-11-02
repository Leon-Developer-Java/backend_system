import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import netCDF4
import numpy as np
from PIL import Image

from adapters import radar_adapter, gridrad_adapter


def gridrad_file(path, stamp="2021-05-14 23:55:00Z", invalid_index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as nc:
        for key, size in (("Longitude", 4), ("Latitude", 3), ("Altitude", 2), ("Index", 8)):
            nc.createDimension(key, size)
        for key, values, units in (("Longitude", [253, 254, 255, 256], "degrees_east"), ("Latitude", [34, 35, 36], "degrees_north"), ("Altitude", [1, 2], "km")):
            var = nc.createVariable(key, "f8", (key,))
            var[:] = values
            var.units = units
        nc.Analysis_time = stamp
        nc.createVariable("index", "i4", ("Index",))[:] = [0, 3, 5, 8, 12, 15, 17, 99 if invalid_index else 20]
        for name, unit, values in (
            ("Reflectivity", "dBZ", [30, 31, 32, 33, 34, 35, 36, 37]),
            ("DifferentialReflectivity", "dB", [-0.5, 0, 1, 2, -0.3, 0.1, 1.1, 2.2]),
            ("DifferentialPhase", "deg km^-1", [-0.2, 0.1, 0.5, 1.2, -0.1, 0.2, 0.7, 1.4]),
            ("CorrelationCoefficient", "dimensionless", [0.97] * 8),
        ):
            var = nc.createVariable(name, "f4", ("Index",), fill_value=np.nan)
            var[:] = values
            var.units = unit
    return path


class GridRadAdapterTest(unittest.TestCase):
    def test_sparse_values_missing_mask_longitude_heights_and_four_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = gridrad_file(root / "sample.nc")
            fields = gridrad_adapter.read_fields(source, {"heights_km": [1]})
            self.assertEqual(fields["lon"].tolist(), [-107, -106, -105, -104])
            self.assertEqual(fields["alt_m"].tolist(), [1000])
            self.assertAlmostEqual(float(fields["fields"]["DifferentialReflectivity"][0, 0, 0]), -0.5)
            self.assertTrue(np.isnan(fields["fields"]["Reflectivity"][0, 0, 1]))
            output = root / "products" / "Radar" / "stage"
            meta = gridrad_adapter.process_file(source, options={"heights_km": [1, 2], "_product_root": str(root / "products")}, output_dir=output)
            self.assertEqual(len(meta["display_products"]), 4)
            self.assertEqual([len(p["levels"]) for p in meta["display_products"]], [3, 2, 2, 2])
            self.assertEqual(meta["spatial"]["level_type"], "height_m_msl")
            self.assertEqual(meta["file_format"], "GRIDRAD_SEVERE_V4_2")
            self.assertTrue(meta["default_webp"].startswith("/data/Radar/"))
            for path in output.glob("*.webp"):
                with Image.open(path) as image:
                    self.assertEqual(image.size, (4, 3))
            with Image.open(output / "DifferentialReflectivity.level-0.webp") as image:
                rgba = np.asarray(image.convert("RGBA"))
                self.assertGreater(rgba[-1, 0, 3], 0)  # South at bottom; negative ZDR is valid.
                self.assertEqual(rgba[-1, 1, 3], 0)  # Unobserved stays transparent.

    def test_series_crosses_utc_day_and_retains_each_field_height_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = gridrad_file(root / "a.nc")
            b = gridrad_file(root / "b.nc", "2021-05-15 00:00:00Z")
            meta = radar_adapter.process_files([str(b), str(a)], gridrad_options={"heights_km": [1], "_product_root": str(root / "products")}, output_dir=root / "products" / "Radar" / "series")
            self.assertEqual(meta["times"], ["2021-05-14T23:55:00Z", "2021-05-15T00:00:00Z"])
            self.assertTrue(all(len(layer["webp_urls"]) == 2 for layer in meta["variable_layers"].values()))
            self.assertEqual(len(meta["webp_files"]), 10)
            self.assertTrue(all(Path(f["meta_file"]).is_file() for f in meta["frames"]))

    def test_invalid_index_nonexistent_height_and_outside_area_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = gridrad_file(root / "bad.nc", invalid_index=True)
            with self.assertRaisesRegex(ValueError, "索引"):
                gridrad_adapter.read_fields(invalid, {})
            valid = gridrad_file(root / "ok.nc")
            for options in ({"heights_km": [1.5]}, {"area": [50, -110, 20, -90]}):
                with self.assertRaises(ValueError):
                    gridrad_adapter.read_fields(valid, options)

    def test_cap_fmt_path_and_legacy_selectable_products_are_unchanged(self):
        self.assertNotIn("observation.base_zdr_cor_log", radar_adapter.RADAR_SELECTABLE_PRODUCTS)
        with patch.object(gridrad_adapter.netCDF4, "Dataset", side_effect=OSError("h5-only fixture")):
            self.assertFalse(gridrad_adapter.is_gridrad("legacy.nc"))
        before = radar_adapter._color_scale_for_product("observation.base_ref_cor_log")[0]
        np.testing.assert_equal(before, radar_adapter.RADAR_THRESHOLDS)


if __name__ == "__main__":
    unittest.main()
