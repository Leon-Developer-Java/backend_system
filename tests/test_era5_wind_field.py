from pathlib import Path
import json
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from adapters import era5_adapter
from services.adapter_runner import cleanup_stage, publish_adapter_output, run_adapter
from workers import adapter_subprocess
from workers.parse_worker import _sync_published_era5_meta


class Era5WindFieldTests(unittest.TestCase):
    def _dataset(
        self,
        *,
        include_v: bool = True,
        v_unit: str = "m s**-1",
        static_v: bool = False,
        longitude: np.ndarray | None = None,
    ) -> xr.Dataset:
        times = np.asarray(["2025-07-01T08:00", "2025-07-01T09:00"], dtype="datetime64[m]")
        latitude = np.asarray([-10.0, 10.0], dtype=np.float64)
        longitude = longitude if longitude is not None else np.asarray([240.0, 120.0, 0.0])
        u10 = np.asarray(
            [
                [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            dtype=np.float32,
        )
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, str]]] = {
            "u10": (
                ("valid_time", "latitude", "longitude"),
                u10,
                {"units": "m s**-1", "long_name": "10 metre U wind component"},
            ),
            "t2m": (
                ("valid_time", "latitude", "longitude"),
                u10 + 273.15,
                {"units": "K", "long_name": "2 metre temperature"},
            ),
        }
        if include_v:
            v10 = np.asarray(
                [
                    [[10.0, 20.0, 30.0], [40.0, np.inf, 60.0]],
                    [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]],
                ],
                dtype=np.float32,
            )
            if static_v:
                data_vars["v10"] = (
                    ("latitude", "longitude"),
                    v10[0],
                    {"units": v_unit, "long_name": "10 metre V wind component"},
                )
            else:
                data_vars["v10"] = (
                    ("valid_time", "latitude", "longitude"),
                    v10,
                    {"units": v_unit, "long_name": "10 metre V wind component"},
                )
        return xr.Dataset(
            data_vars=data_vars,
            coords={"valid_time": times, "latitude": latitude, "longitude": longitude},
        )

    def _process(self, dataset: xr.Dataset, directory: Path) -> tuple[Path, dict]:
        source = directory / "wind.nc"
        dataset.to_netcdf(source, engine="netcdf4")
        with (
            patch.object(era5_adapter, "TARGET_RESOLUTIONS_KM", ()),
            patch("services.era5_store.sync_meta", return_value={"status": "test"}),
        ):
            meta = era5_adapter.process_file(str(source))
        return source, meta

    def test_generates_paired_float32_frames_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, meta = self._process(self._dataset(), root)

            wind = meta["wind_field"]
            self.assertTrue(wind["available"])
            self.assertEqual(wind["components"], {"u": "u10", "v": "v10"})
            self.assertEqual(wind["level"], "10 m above ground")
            self.assertEqual(wind["unit"], "m/s")
            self.assertEqual(wind["source_units"], {"u": "m s**-1", "v": "m s**-1"})
            self.assertEqual(wind["times"], ["2025-07-01T08:00", "2025-07-01T09:00"])
            self.assertEqual(wind["grid"]["width"], 3)
            self.assertEqual(wind["grid"]["height"], 2)
            self.assertEqual(wind["grid"]["extent"], [-120.0, -10.0, 120.0, 10.0])
            self.assertEqual(wind["grid"]["row_order"], "north_to_south")
            self.assertEqual(wind["grid"]["column_order"], "west_to_east")
            self.assertEqual(wind["grid"]["lon_step"], 120.0)
            self.assertEqual(wind["grid"]["lat_step"], 20.0)
            self.assertTrue(wind["grid"]["periodic_longitude"])
            self.assertEqual(wind["encoding"]["dtype"], "float32")
            self.assertEqual(wind["encoding"]["byte_order"], "little")
            self.assertEqual(len(wind["frames"]), 2)
            self.assertEqual(wind["speed_variable"], "ws10")
            self.assertEqual(wind["display_range"], {"min": 0.0, "max": 125.0})
            self.assertEqual(wind["palette"], list(era5_adapter.WIND_SPEED_PALETTE))

            expected_size = 3 * 2 * 4
            for index, frame in enumerate(wind["frames"]):
                self.assertEqual(frame["index"], index)
                self.assertEqual(frame["component_byte_length"], expected_size)
                self.assertGreaterEqual(frame["speed_min"], 0)
                self.assertLessEqual(frame["speed_max"], wind["display_range"]["max"])
                self.assertEqual((root / f"{source.stem}_u10_step{index:03d}.float32").stat().st_size, expected_size)
                self.assertEqual((root / f"{source.stem}_v10_step{index:03d}.float32").stat().st_size, expected_size)
                self.assertTrue((root / Path(frame["speed_webp_url"]).name).is_file())

            u_values = np.fromfile(root / "wind_u10_step000.float32", dtype="<f4")
            v_values = np.fromfile(root / "wind_v10_step000.float32", dtype="<f4")
            np.testing.assert_array_equal(
                u_values,
                np.asarray([4.0, 6.0, era5_adapter.NODATA, 1.0, era5_adapter.NODATA, 2.0], dtype="<f4"),
            )
            np.testing.assert_array_equal(
                v_values,
                np.asarray([40.0, 60.0, era5_adapter.NODATA, 10.0, era5_adapter.NODATA, 20.0], dtype="<f4"),
            )

            variables = {item["name"]: item for item in meta["variables"]}
            self.assertIn("ws10", variables)
            self.assertEqual(len(variables["u10"]["float32"]["paths"]), 2)
            self.assertEqual(len(variables["v10"]["float32"]["paths"]), 2)
            self.assertTrue(variables["ws10"]["derived"])
            self.assertEqual(variables["ws10"]["derived_from"], ["u10", "v10"])
            self.assertGreaterEqual(variables["ws10"]["stats"]["min"], 0)
            self.assertEqual(variables["ws10"]["float32"]["paths"], [])
            self.assertEqual(len(meta["variable_layers"]["u10"]["float32_urls"]), 2)
            self.assertEqual(
                meta["variable_layers"]["u10"]["resolution_layers"]["native"]["float32_urls"],
                meta["variable_layers"]["u10"]["float32_urls"],
            )
            speed_layer = meta["variable_layers"]["ws10"]
            self.assertEqual(len(speed_layer["webp_urls"]), 2)
            self.assertEqual(speed_layer["display_range"], wind["display_range"])
            self.assertEqual(speed_layer["palette"], wind["palette"])
            self.assertEqual(
                speed_layer["resolution_layers"]["native"]["display_range"],
                wind["display_range"],
            )
            self.assertTrue(meta["webp_files"])
            self.assertTrue(all((root / Path(path).name).exists() for path in meta["webp_files"]))

    def test_missing_component_degrades_without_breaking_webp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, meta = self._process(self._dataset(include_v=False), root)

            self.assertFalse(meta["wind_field"]["available"])
            self.assertEqual(meta["wind_field"]["reason"], "missing_components")
            self.assertEqual(meta["wind_field"]["detail"]["missing"], ["v"])
            self.assertTrue(meta["webp_files"])
            self.assertFalse(list(root.glob("*.float32")))
            self.assertNotIn("ws10", meta["variable_layers"])
            u_meta = next(item for item in meta["variables"] if item["name"] == "u10")
            self.assertEqual(u_meta["float32"]["paths"], [])

    def test_incompatible_time_or_units_degrades_safely(self) -> None:
        cases = (
            (self._dataset(static_v=True), "time_dimension_mismatch"),
            (self._dataset(v_unit="knots"), "incompatible_units"),
        )
        for dataset, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    _, meta = self._process(dataset, root)
                    self.assertFalse(meta["wind_field"]["available"])
                    self.assertEqual(meta["wind_field"]["reason"], expected_reason)
                    self.assertTrue(meta["webp_files"])
                    self.assertFalse(list(root.glob("*.float32")))

    def test_auxiliary_valid_time_uses_shared_time_dimension(self) -> None:
        dataset = self._dataset().rename({"valid_time": "time"})
        valid_times = np.asarray(
            ["2025-07-01T08:00", "2025-07-01T09:00"],
            dtype="datetime64[m]",
        )
        dataset = dataset.assign_coords(
            time=np.asarray([0, 1], dtype=np.int32),
            valid_time=("time", valid_times),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, meta = self._process(dataset, root)

            self.assertTrue(meta["wind_field"]["available"])
            self.assertEqual(
                meta["wind_field"]["times"],
                ["2025-07-01T08:00", "2025-07-01T09:00"],
            )
            self.assertNotEqual(
                meta["variable_layers"]["u10"]["stats"][0]["mean"],
                meta["variable_layers"]["u10"]["stats"][1]["mean"],
            )
            second_frame = np.fromfile(root / "wind_u10_step001.float32", dtype="<f4")
            np.testing.assert_array_equal(
                second_frame,
                np.asarray([10.0, 12.0, 11.0, 7.0, 9.0, 8.0], dtype="<f4"),
            )

    def test_scalar_valid_time_does_not_hide_time_dimension(self) -> None:
        dataset = self._dataset().rename({"valid_time": "time"})
        dataset = dataset.assign_coords(valid_time=np.datetime64("2025-07-01T08:00"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, meta = self._process(dataset, root)

            self.assertTrue(meta["wind_field"]["available"])
            self.assertEqual(len(meta["times"]), 2)
            self.assertEqual(meta["wind_field"]["times"], meta["times"])
            self.assertEqual(len(meta["wind_field"]["frames"]), 2)

    def test_duplicate_longitude_after_normalization_is_rejected(self) -> None:
        longitude = np.asarray([0.0, 180.0, 360.0], dtype=np.float64)
        dataset = self._dataset(longitude=longitude)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "wind.nc"
            dataset.to_netcdf(source, engine="netcdf4")
            with self.assertRaisesRegex(era5_adapter.Era5ValidationError, "duplicate values"):
                era5_adapter.process_file(str(source))
            self.assertFalse(list(root.glob("*.float32")))
            self.assertFalse(list(root.glob("*.webp")))

    def test_empty_and_corrupt_sources_are_rejected_without_outputs(self) -> None:
        for content, expected_message in ((b"", "empty"), (b"not-netcdf", "could not be opened")):
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    source = root / "broken.nc"
                    source.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, expected_message):
                        era5_adapter.process_file(str(source))
                    self.assertEqual(list(root.iterdir()), [source])

    def test_missing_key_variable_is_rejected_before_asset_generation(self) -> None:
        dataset = self._dataset().rename({"t2m": "custom_temperature"}).drop_vars(["u10", "v10"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "unsupported.nc"
            dataset.to_netcdf(source, engine="netcdf4")
            with self.assertRaisesRegex(era5_adapter.Era5ValidationError, "no supported key variable"):
                era5_adapter.process_file(str(source))
            self.assertFalse(list(root.glob("*.webp")))

    def test_db_sync_failure_stops_success_and_cleans_generated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "wind.nc"
            self._dataset().to_netcdf(source, engine="netcdf4")
            with (
                patch.object(era5_adapter, "TARGET_RESOLUTIONS_KM", ()),
                patch("services.era5_store.sync_meta", side_effect=RuntimeError("db unavailable")),
                self.assertRaisesRegex(RuntimeError, "db unavailable"),
            ):
                era5_adapter.process_file(str(source))
            self.assertFalse(list(root.glob("*.webp")))
            self.assertFalse(list(root.glob("*.float32")))
            self.assertFalse(list(root.glob("*.meta.json")))

    def test_worker_defers_era5_index_sync_until_assets_are_published(self) -> None:
        from services import era5_store

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw" / "wind.nc"
            raw.parent.mkdir()
            self._dataset().to_netcdf(raw, engine="netcdf4")
            product_root = root / "products"
            stage = product_root / "ERA5" / ".adapter_staging" / "file-1" / "attempt-1"

            with (
                patch.object(era5_adapter, "TARGET_RESOLUTIONS_KM", ()),
                patch("services.era5_store.sync_meta", side_effect=AssertionError("synced before publish")),
            ):
                child = run_adapter(
                    file_uuid="file-1",
                    data_type="ERA5",
                    source_path=raw,
                    output_root=product_root,
                    attempt_dir=stage,
                    original_file_name="wind.nc",
                )

            meta, meta_file, final_dir = publish_adapter_output(child, product_root)
            self.assertTrue(final_dir.is_dir())
            self.assertNotIn("db_sync", meta["extra"]["era5"])

            index_dir = root / "index"
            with patch.object(era5_store, "DATA_DIR", index_dir):
                _sync_published_era5_meta(meta, meta_file)

            stored = json.loads(meta_file.read_text(encoding="utf-8"))
            self.assertIn("db_sync", stored["extra"]["era5"])
            self.assertTrue((index_dir / era5_store.DB_NAME).is_file())
            self.assertTrue(all("/assets/file-1/" in value for value in stored["webp_files"]))

    def test_corrupt_worker_input_is_non_retryable_and_staging_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "broken.nc"
            raw.write_bytes(b"not-a-netcdf")
            product_root = root / "products"
            stage = product_root / "ERA5" / ".adapter_staging" / "file-2" / "attempt-1"
            job = {
                "file_uuid": "file-2",
                "collection_uuid": None,
                "data_type": "ERA5",
                "source_path": raw.as_posix(),
                "original_file_name": raw.name,
                "output_root": product_root.as_posix(),
                "stage_dir": stage.as_posix(),
                "result_path": (root / "result.json").as_posix(),
                "error_path": (root / "error.json").as_posix(),
            }
            job_path = root / "job.json"
            job_path.write_text(json.dumps(job), encoding="utf-8")

            with patch.object(sys, "argv", ["adapter_subprocess", "--job", str(job_path)]):
                exit_code = adapter_subprocess.main()

            error = json.loads((root / "error.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(error["error_type"], "ValueError")
            self.assertFalse(error["retryable"])
            self.assertFalse((root / "result.json").exists())
            cleanup_stage(stage, product_root)
            self.assertFalse(stage.exists())

    def test_float32_write_failure_degrades_and_cleans_partial_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, _ = self._process(self._dataset(), root)
            existing_assets = {
                path: path.read_bytes()
                for path in root.glob("*.float32")
            }
            original_writer = era5_adapter._stage_float32
            call_count = 0

            def fail_on_second_write(data: np.ndarray, output_path: Path) -> Path:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("synthetic write failure")
                return original_writer(data, output_path)

            with (
                patch.object(era5_adapter, "TARGET_RESOLUTIONS_KM", ()),
                patch.object(era5_adapter, "_stage_float32", side_effect=fail_on_second_write),
                patch("services.era5_store.sync_meta", return_value={"status": "test"}),
            ):
                meta = era5_adapter.process_file(str(source))

            self.assertFalse(meta["wind_field"]["available"])
            self.assertEqual(meta["wind_field"]["reason"], "wind_field_generation_failed")
            self.assertTrue(meta["webp_files"])
            self.assertEqual(
                {path: path.read_bytes() for path in root.glob("*.float32")},
                existing_assets,
            )
            self.assertFalse(list(root.glob(".*.tmp")))
            self.assertFalse(list(root.glob(".*.backup")))

    def test_optional_wind_field_keeps_existing_sqlite_schema_compatible(self) -> None:
        from services import era5_store

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, meta = self._process(self._dataset(), root)
            db_path = root / "era5_index.db"

            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            try:
                era5_store.init_db(connection)
                result = era5_store._upsert_meta(connection, meta)
                connection.commit()
                stored_counts = {
                    "datasets": connection.execute("SELECT COUNT(*) FROM era5_dataset").fetchone()[0],
                    "variables": connection.execute("SELECT COUNT(*) FROM era5_variable").fetchone()[0],
                    "time_steps": connection.execute("SELECT COUNT(*) FROM era5_time_step").fetchone()[0],
                    "assets": connection.execute("SELECT COUNT(*) FROM era5_layer_asset").fetchone()[0],
                }
                asset_urls = [
                    row[0]
                    for row in connection.execute("SELECT webp_url FROM era5_layer_asset").fetchall()
                ]
            finally:
                connection.close()

            self.assertEqual(result["datasets"], 1)
            self.assertEqual(result["variables"], len(meta["variables"]))
            self.assertEqual(result["time_steps"], len(meta["times"]))
            self.assertEqual(result["layer_assets"], len(meta["webp_files"]))
            self.assertEqual(stored_counts["datasets"], 1)
            self.assertEqual(stored_counts["variables"], len(meta["variables"]))
            self.assertEqual(stored_counts["time_steps"], len(meta["times"]))
            self.assertEqual(stored_counts["assets"], len(meta["webp_files"]))
            self.assertTrue(all(url.endswith(".webp") for url in asset_urls))


if __name__ == "__main__":
    unittest.main()
