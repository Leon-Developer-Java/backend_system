from pathlib import Path
import unittest

from adapters import radar_adapter


class RadarAdapterPathTests(unittest.TestCase):
    def test_webp_path_does_not_repeat_long_radar_source_name(self) -> None:
        source = Path(
            "D:/Study/weatherProject/backend_system/data/Radar/.adapter_staging/"
            "73d73736-64c0-4ae8-a2d6-a634268de7f5/"
            "d2c15910-4a79-4eaf-b729-aeef8507722a/"
            "Z_RADR_I_BJSCN_20250601133000_O_DOR_MOC_CAP_FMT.nc"
        )

        output = radar_adapter._webp_output_path(
            source,
            "observation.base_ref_cor_log",
            "max",
        )

        self.assertEqual(
            output.name,
            "observation.base_ref_cor_log.max.webp",
        )
        self.assertEqual(output.parent.name, source.stem)
        self.assertNotIn(source.stem, output.name)
        self.assertLess(len(str(output)), 260)

    def test_webp_file_name_sanitizes_variable_and_level(self) -> None:
        self.assertEqual(
            radar_adapter._webp_file_name("observation/rate", "level 0"),
            "observation_rate.level_0.webp",
        )


if __name__ == "__main__":
    unittest.main()
