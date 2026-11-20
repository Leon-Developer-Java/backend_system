import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from eccodes import codes_grib_new_from_samples, codes_release, codes_set, codes_set_values, codes_write

from adapters.seasonal_precip_adapter import _convert, _infer_source_metadata, process_file
from services.adapter_runner import canonical_data_type, publish_adapter_output, run_adapter
from services.asset_catalog import build_asset_catalog


class SeasonalPrecipAdapterTest(unittest.TestCase):
    def test_cpc_month_coordinates_and_fractional_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'prate.202610.prob.seas.nc'
            shape = (2, 2, 2)
            dataset = xr.Dataset({
                'prob_above': (('target','lat','lon'), np.full(shape,0.6), {'units':'percent'}),
                'prob_norm': (('target','lat','lon'), np.full(shape,0.3), {'units':'percent'}),
                'prob_below': (('target','lat','lon'), np.full(shape,0.1), {'units':'percent'}),
            }, coords={'target': ('target',[801.,802.],{'units':'months since 1960-01-01 12:00:00'}),'lat':[1.,0.],'lon':[0.,1.]})
            dataset.to_netcdf(source,engine='scipy')
            result = process_file(str(source))
            layer = result['variable_layers']['prob_above']
            self.assertTrue(layer['times'][0].startswith('2026-10-01'))
            self.assertEqual(layer['mean'],60.)
            self.assertEqual(layer['unit'],'%')
            self.assertIn('NMME',result['product'])

    def test_current_model_anomaly_identity_and_rate_units(self):
        product,provider,issue = _infer_source_metadata(Path('CanESM5.prate.202610.ENSMEAN.anom.nc'),{})
        self.assertIn('CanESM5',product)
        self.assertEqual(provider,'NOAA/CPC')
        self.assertEqual(issue,'2026-10-01T00:00:00+00:00')
        values,unit,_ = _convert(np.array([1.0]),'mm/s','anomaly','fcst')
        self.assertEqual(unit,'mm/day')
        self.assertEqual(values[0],86400.)
        values,unit,_ = _convert(np.array([1.0]),'m s**-1','anomaly','tpara')
        self.assertEqual(values[0],86400000.)
        self.assertEqual(unit,'mm/day')

    def test_percent_encoding_is_preserved_for_small_probabilities(self):
        values,unit,_ = _convert(np.array([0.6]),'percent','probability','prob_above')
        self.assertEqual(values[0],0.6)

    def test_cancm4_filename_preserves_product_and_issue_date(self):
        product, provider, issue = _infer_source_metadata(Path('pr_daily_cancm4.20230701_E2_20230701-20240630.nc'), {})
        self.assertEqual(product, 'NOAA NMME CanCM4')
        self.assertEqual(provider, 'NOAA/NCEI')
        self.assertEqual(issue, '2023-07-01T00:00:00+00:00')

    def test_data_type_aliases_share_one_worker_contract(self):
        for value in ("SEASONAL_PRECIP", "SeasonalPrecip", "seasonal_precipitation", "季节降水"):
            self.assertEqual(canonical_data_type(value), "SEASONAL_PRECIP")

    def test_netcdf_builds_amount_anomaly_and_probability_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "seas5_demo.nc"
            shape = (2, 3, 4)
            dataset = xr.Dataset(
                {
                    "precip_total": (
                        ("time", "lat", "lon"),
                        np.arange(24, dtype=float).reshape(shape),
                        {"units": "mm", "long_name": "Seasonal precipitation"},
                    ),
                    "precip_anomaly": (
                        ("time", "lat", "lon"),
                        np.linspace(-5, 5, 24).reshape(shape),
                        {"units": "mm", "long_name": "Precipitation anomaly"},
                    ),
                    "prob_above": (
                        ("time", "lat", "lon"),
                        np.full(shape, 0.25),
                        {"units": "1", "long_name": "Probability above normal"},
                    ),
                },
                coords={
                    "time": np.array(["2026-06-01", "2026-07-01"], dtype="datetime64[ns]"),
                    "lat": [40.0, 30.0, 20.0],
                    "lon": [100.0, 110.0, 120.0, 130.0],
                },
                attrs={"title": "Demo Seasonal Product", "provider": "Demo"},
            )
            dataset.to_netcdf(source, engine="netcdf4")

            meta = process_file(str(source))

            self.assertEqual(meta["data_type"], "SEASONAL_PRECIP")
            self.assertEqual(meta["product"], "Demo Seasonal Product")
            self.assertEqual(meta["default_variable"], "precip_total")
            self.assertEqual(meta["extent"], [100.0, 20.0, 130.0, 40.0])
            self.assertEqual(meta["variable_layers"]["prob_above"]["unit"], "%")
            self.assertEqual(len(meta["variable_layers"]["precip_total"]["frames"]), 2)
            self.assertTrue(all(Path(path).is_file() for path in source.parent.glob("*.webp")))

            product_root = Path(temporary) / "products"
            stage_dir = product_root / "SeasonalPrecip" / ".adapter_staging" / "demo" / "attempt"
            child_result = run_adapter(
                "demo-file",
                "SeasonalPrecip",
                source,
                product_root,
                stage_dir,
                original_file_name=source.name,
            )
            published_meta, _meta_file, final_dir = publish_adapter_output(child_result, product_root)
            assets = build_asset_catalog(
                file_uuid="demo-file",
                data_type="SEASONALPRECIP",
                meta=published_meta,
                product_root=product_root,
                final_dir=final_dir,
            )
            self.assertEqual(final_dir.name, "demo-file")
            self.assertEqual(len(assets), 6)
            self.assertEqual(assets[0]["product_name"], "Demo Seasonal Product")
            self.assertEqual(assets[0]["provider"], "Demo")

    def test_probability_category_dimension_becomes_separate_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "category_product.nc"
            values = np.full((1, 3, 2, 2), 1 / 3, dtype=float)
            dataset = xr.Dataset(
                {
                    "precip_probability": (
                        ("time", "category", "lat", "lon"),
                        values,
                        {"units": "1", "long_name": "Precipitation category probability"},
                    ),
                },
                coords={
                    "time": np.array(["2026-06-01"], dtype="datetime64[ns]"),
                    "category": ["below", "normal", "above"],
                    "lat": [40.0, 20.0],
                    "lon": [100.0, 120.0],
                },
            )
            dataset.to_netcdf(source, engine="netcdf4")

            meta = process_file(str(source))

            self.assertEqual(
                set(meta["variable_layers"]),
                {"precip_probability_below", "precip_probability_normal", "precip_probability_above"},
            )
            self.assertTrue(all(item["varType"] == "probability" for item in meta["variable_layers"].values()))

    def test_grib2_precipitation_product_is_rendered(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "seasonal_tp.grib2"
            handle = codes_grib_new_from_samples("regular_ll_sfc_grib2")
            try:
                for key, value in (
                    ("Ni", 4),
                    ("Nj", 3),
                    ("latitudeOfFirstGridPointInDegrees", 40.0),
                    ("latitudeOfLastGridPointInDegrees", 20.0),
                    ("longitudeOfFirstGridPointInDegrees", 100.0),
                    ("longitudeOfLastGridPointInDegrees", 130.0),
                    ("shortName", "tp"),
                    ("stepType", "accum"),
                    ("dataDate", 20260601),
                    ("dataTime", 0),
                    ("step", 3),
                ):
                    codes_set(handle, key, value)
                codes_set_values(handle, np.arange(12, dtype=float))
                with source.open("wb") as output:
                    codes_write(handle, output)
            finally:
                codes_release(handle)

            meta = process_file(str(source))

            self.assertEqual(meta["default_variable"], "tp")
            self.assertEqual(meta["variable_layers"]["tp"]["varType"], "precipitation")
            self.assertEqual(len(meta["variable_layers"]["tp"]["frames"]), 1)


if __name__ == "__main__":
    unittest.main()
