from pathlib import Path
import tempfile
import unittest

from sqlalchemy import create_engine, insert

from services import source_resolver
from DB.schema import metadata, public_info


class SourceResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_root = Path(self.temp_dir.name) / "raw"
        self.raw_file = self.raw_root / "user_upload" / "sample.nc"
        self.raw_file.parent.mkdir(parents=True)
        self.raw_file.write_bytes(b"nc")
        self.file_uuid = "11111111-2222-4333-8444-555555555555"

        self.engine = create_engine("sqlite:///:memory:")
        metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(
                insert(public_info).values(
                    file_uuid=self.file_uuid,
                    acquisition_type="user_upload",
                    visibility="private",
                    data_type="CMA",
                    file_type="nc",
                    original_file_name="sample.nc",
                    stored_file_name="sample.nc",
                    source_path="user_upload/sample.nc",
                    file_size=2,
                    file_hash="0" * 64,
                    ingest_status="success",
                    parse_status="success",
                )
            )

        self.original_engine = source_resolver._ENGINE
        self.original_root = source_resolver.RAW_STORAGE_ROOT
        source_resolver._ENGINE = self.engine
        source_resolver.RAW_STORAGE_ROOT = self.raw_root.resolve()

    def tearDown(self) -> None:
        source_resolver._ENGINE = self.original_engine
        source_resolver.RAW_STORAGE_ROOT = self.original_root
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_resolves_raw_file_from_published_asset_uuid(self) -> None:
        published_hint = Path("CMA") / "assets" / self.file_uuid / "sample.nc"

        resolved = source_resolver.resolve_raw_source(published_hint, "CMA")

        self.assertEqual(resolved, self.raw_file.resolve())

    def test_rejects_a_different_data_type(self) -> None:
        published_hint = Path("CMA") / "assets" / self.file_uuid / "sample.nc"

        self.assertIsNone(source_resolver.resolve_raw_source(published_hint, "ERA5"))


if __name__ == "__main__":
    unittest.main()
