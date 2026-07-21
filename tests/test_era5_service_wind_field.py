from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services import era5_service


class Era5ServiceWindFieldTests(unittest.TestCase):
    def _base_meta(self) -> dict:
        times = ["2025-07-01T08:00", "2025-07-01T09:00"]
        return {
            "schema_version": "1.0",
            "dataset_id": "wind_nc",
            "data_type": "ERA5",
            "default_variable": "t2m",
            "times": times,
            "extent": [-180.0, -90.0, 179.75, 90.0],
            "available_resolutions": ["native"],
            "variable_options": [{"name": "t2m", "label": "temperature", "unit": "K"}],
            "variables": [{"name": "t2m", "long_name": "temperature", "unit": "K"}],
            "variable_layers": {
                "t2m": {
                    "name": "t2m",
                    "label": "temperature",
                    "unit": "K",
                    "width": 2,
                    "height": 2,
                    "extent": [-180.0, -90.0, 179.75, 90.0],
                    "times": times,
                    "webp_urls": [],
                    "image_urls": [],
                    "stats": [
                        {"min": 270.0, "max": 280.0, "mean": 275.0},
                        {"min": 271.0, "max": 281.0, "mean": 276.0},
                    ],
                    "nodata": -999999.0,
                    "resolution": "native",
                    "available_resolutions": ["native"],
                    "resolution_layers": {},
                    "resolution_status": {},
                }
            },
        }

    def _available_wind(self, frames: list[dict]) -> dict:
        times = ["2025-07-01T08:00", "2025-07-01T09:00"]
        return {
            "schema_version": "1.0",
            "available": True,
            "product": "10m_wind",
            "components": {"u": "u10", "v": "v10"},
            "level": "10 m above ground",
            "unit": "m/s",
            "times": times,
            "grid": {
                "crs": "EPSG:4326",
                "width": 2,
                "height": 2,
                "extent": [-180.0, -90.0, 179.75, 90.0],
                "origin": "north_west",
                "row_order": "north_to_south",
                "column_order": "west_to_east",
            },
            "encoding": {
                "dtype": "float32",
                "byte_order": "little",
                "layout": "component_separated",
                "array_order": "C",
                "bytes_per_value": 4,
                "nodata": -999999.0,
            },
            "frames": frames,
        }

    def _add_speed_contract(self, meta: dict) -> None:
        palette = ["#2563eb", "#0891b2", "#16a34a", "#facc15", "#dc2626"]
        display_range = {"min": 0.0, "max": 30.0}
        meta["variable_layers"]["ws10"] = {
            "name": "ws10",
            "label": "10 metre wind speed",
            "unit": "m/s",
            "display_unit": "m/s",
            "times": list(meta["times"]),
            "display_range": display_range,
            "palette": palette,
        }
        meta["wind_field"].update({
            "speed_variable": "ws10",
            "display_range": display_range,
            "palette": palette,
        })

    def _display(self, meta: dict, data_dir: Path) -> dict:
        with (
            patch.object(era5_service, "DATA_DIR", data_dir),
            patch.object(era5_service, "_latest_meta", return_value=meta),
        ):
            return era5_service.get_display_data()

    def test_available_wind_is_validated_and_urls_are_public(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "ERA5"
            data_dir.mkdir()
            for name in ("u0.float32", "v0.float32", "u1.float32", "v1.float32"):
                (data_dir / name).write_bytes(b"\x00" * 16)

            meta = self._base_meta()
            meta["wind_field"] = self._available_wind([
                {
                    "index": 0,
                    "time": "2025-07-01T08:00",
                    "u_url": str(data_dir / "u0.float32"),
                    "v_url": "/srv/runtime/data/ERA5/v0.float32",
                    "component_byte_length": 16,
                },
                {
                    "index": 1,
                    "time": "2025-07-01T09:00",
                    "u_url": "/data/ERA5/u1.float32",
                    "v_url": "v1.float32",
                    "component_byte_length": 16,
                },
            ])
            self._add_speed_contract(meta)
            original = deepcopy(meta)

            result = self._display(meta, data_dir)

            wind = result["wind_field"]
            self.assertTrue(wind["available"])
            self.assertEqual(wind["times"], meta["times"])
            self.assertEqual(wind["grid"], meta["wind_field"]["grid"])
            self.assertEqual(wind["encoding"], meta["wind_field"]["encoding"])
            self.assertEqual(wind["speed_variable"], "ws10")
            self.assertEqual(wind["display_range"], {"min": 0.0, "max": 30.0})
            self.assertEqual(wind["palette"], meta["variable_layers"]["ws10"]["palette"])
            self.assertEqual(
                [(frame["u_url"], frame["v_url"]) for frame in wind["frames"]],
                [
                    ("/data/ERA5/u0.float32", "/data/ERA5/v0.float32"),
                    ("/data/ERA5/u1.float32", "/data/ERA5/v1.float32"),
                ],
            )
            self.assertEqual(result["meta_json"]["wind_field"], wind)
            self.assertEqual(meta, original)

    def test_speed_style_mismatch_disables_only_particles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "ERA5"
            data_dir.mkdir()
            for name in ("u0.float32", "v0.float32", "u1.float32", "v1.float32"):
                (data_dir / name).write_bytes(b"\x00" * 16)
            meta = self._base_meta()
            meta["wind_field"] = self._available_wind([
                {
                    "index": index,
                    "time": meta["times"][index],
                    "u_url": f"/data/ERA5/u{index}.float32",
                    "v_url": f"/data/ERA5/v{index}.float32",
                    "component_byte_length": 16,
                }
                for index in range(2)
            ])
            self._add_speed_contract(meta)
            meta["variable_layers"]["ws10"]["display_range"] = {"min": 0.0, "max": 25.0}

            result = self._display(meta, data_dir)

            self.assertFalse(result["wind_field"]["available"])
            self.assertEqual(result["wind_field"]["detail"], {"code": "speed_style_mismatch"})
            self.assertIn("ws10", result["variable_layers"])

    def test_old_meta_without_wind_field_remains_displayable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "ERA5"
            data_dir.mkdir()
            meta = self._base_meta()

            result = self._display(meta, data_dir)

            self.assertFalse(result["wind_field"]["available"])
            self.assertEqual(result["wind_field"]["reason"], "not_provided")
            self.assertEqual(result["variable_layers"], meta["variable_layers"])
            self.assertEqual(result["times"], meta["times"])
            self.assertEqual(result["meta_json"]["wind_field"], result["wind_field"])

    def test_adapter_unavailable_reason_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "ERA5"
            data_dir.mkdir()
            meta = self._base_meta()
            meta["wind_field"] = {
                "schema_version": "1.0",
                "available": False,
                "product": "10m_wind",
                "components": {"u": "u10", "v": None},
                "reason": "missing_components",
                "detail": {"missing": ["v"]},
            }

            result = self._display(meta, data_dir)

            self.assertFalse(result["wind_field"]["available"])
            self.assertEqual(result["wind_field"]["reason"], "missing_components")
            self.assertEqual(result["wind_field"]["detail"], {"missing": ["v"]})

    def test_unsafe_asset_path_degrades_without_breaking_static_display(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "ERA5"
            data_dir.mkdir()
            (data_dir / "v0.float32").write_bytes(b"\x00" * 16)
            meta = self._base_meta()
            frames = [
                {
                    "index": 0,
                    "time": "2025-07-01T08:00",
                    "u_url": "https://example.invalid/data/ERA5/u0.float32",
                    "v_url": "/data/ERA5/v0.float32",
                    "component_byte_length": 16,
                },
                {
                    "index": 1,
                    "time": "2025-07-01T09:00",
                    "u_url": "/data/ERA5/u1.float32",
                    "v_url": "/data/ERA5/v1.float32",
                    "component_byte_length": 16,
                },
            ]
            meta["wind_field"] = self._available_wind(frames)
            original = deepcopy(meta)

            result = self._display(meta, data_dir)

            self.assertFalse(result["wind_field"]["available"])
            self.assertEqual(result["wind_field"]["reason"], "display_contract_invalid")
            self.assertEqual(
                result["wind_field"]["detail"],
                {"code": "asset_url_not_local", "frame_index": 0},
            )
            self.assertEqual(result["variable_layers"], meta["variable_layers"])
            self.assertEqual(meta, original)

            malformed = deepcopy(meta)
            malformed["wind_field"]["frames"][0]["u_url"] = "http://["
            malformed_result = self._display(malformed, data_dir)
            self.assertFalse(malformed_result["wind_field"]["available"])
            self.assertEqual(
                malformed_result["wind_field"]["detail"],
                {"code": "asset_url_invalid", "frame_index": 0},
            )

    def test_missing_or_wrong_sized_asset_degrades_whole_wind_field(self) -> None:
        cases = (
            ("asset_missing", 16, False, 16),
            ("asset_byte_length_mismatch", 12, True, 16),
            ("frame_byte_length_invalid", 16, True, 12),
        )
        for expected_code, byte_count, create_u, declared_byte_length in cases:
            with self.subTest(expected_code=expected_code):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    data_dir = Path(temporary_directory) / "ERA5"
                    data_dir.mkdir()
                    if create_u:
                        (data_dir / "u0.float32").write_bytes(b"\x00" * byte_count)
                    (data_dir / "v0.float32").write_bytes(b"\x00" * 16)
                    meta = self._base_meta()
                    frames = [
                        {
                            "index": 0,
                            "time": "2025-07-01T08:00",
                            "u_url": "/data/ERA5/u0.float32",
                            "v_url": "/data/ERA5/v0.float32",
                            "component_byte_length": declared_byte_length,
                        },
                        {
                            "index": 1,
                            "time": "2025-07-01T09:00",
                            "u_url": "/data/ERA5/u1.float32",
                            "v_url": "/data/ERA5/v1.float32",
                            "component_byte_length": 16,
                        },
                    ]
                    meta["wind_field"] = self._available_wind(frames)
                    original = deepcopy(meta)

                    result = self._display(meta, data_dir)

                    self.assertFalse(result["wind_field"]["available"])
                    self.assertEqual(result["wind_field"]["detail"]["code"], expected_code)
                    self.assertEqual(result["wind_field"]["detail"]["frame_index"], 0)
                    self.assertEqual(meta, original)

    def test_file_race_or_infinite_dimensions_cannot_break_static_display(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "ERA5"
            data_dir.mkdir()
            frames = [
                {
                    "index": 0,
                    "time": "2025-07-01T08:00",
                    "u_url": "/data/ERA5/u0.float32",
                    "v_url": "/data/ERA5/v0.float32",
                    "component_byte_length": 16,
                },
                {
                    "index": 1,
                    "time": "2025-07-01T09:00",
                    "u_url": "/data/ERA5/u1.float32",
                    "v_url": "/data/ERA5/v1.float32",
                    "component_byte_length": 16,
                },
            ]
            meta = self._base_meta()
            meta["wind_field"] = self._available_wind(frames)
            with (
                patch.object(era5_service, "DATA_DIR", data_dir),
                patch.object(Path, "is_file", return_value=True),
                patch.object(Path, "stat", side_effect=PermissionError("synthetic race")),
            ):
                unreadable = era5_service._display_wind_field(meta)
            self.assertFalse(unreadable["available"])
            self.assertEqual(unreadable["detail"], {"code": "asset_unreadable", "frame_index": 0})

            meta["wind_field"]["grid"]["width"] = float("inf")
            with patch.object(era5_service, "DATA_DIR", data_dir):
                invalid_number = era5_service._display_wind_field(meta)
            self.assertFalse(invalid_number["available"])
            self.assertEqual(invalid_number["detail"], {"code": "grid_width_invalid"})

    def test_service_reads_latest_meta_file_and_adds_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "ERA5"
            data_dir.mkdir()
            meta = self._base_meta()
            meta_path = data_dir / "fixture.nc.meta.json"
            with meta_path.open("w", encoding="utf-8") as stream:
                json.dump(meta, stream)

            with patch.object(era5_service, "DATA_DIR", data_dir):
                result = era5_service.get_display_data()

            self.assertEqual(result["meta_json"]["dataset_id"], meta["dataset_id"])
            self.assertFalse(result["wind_field"]["available"])
            self.assertEqual(result["wind_field"]["reason"], "not_provided")

    def test_era5_route_keeps_wind_field_inside_standard_envelope(self) -> None:
        import main as backend_main

        payload = {
            "business_type": "ERA5",
            "wind_field": {"available": False, "reason": "not_provided"},
        }
        with patch.object(
            backend_main.DISPLAY_SERVICES["ERA5"],
            "get_display_data",
            return_value=payload,
        ):
            result = backend_main.display_data(
                business_type="ERA5",
                variable=None,
                level_index=0,
                time_index=0,
                resolution="native",
                meta_file=None,
                scene_id=None,
            )

        self.assertEqual(result["code"], 0)
        self.assertEqual(result["data"]["wind_field"], payload["wind_field"])


if __name__ == "__main__":
    unittest.main()
