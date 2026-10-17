"""Published WRF URLs must not leak unlisted run products."""

import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, insert

from services.wrf_asset_access import allow_wrf_asset
from services import source_resolver
from DB.schema import metadata, public_info, wrf_info


TASK_ID = "wrf_gfs_20260924T000000Z_1234abcd"


class WrfAssetAccessTest(unittest.TestCase):
    def test_workbench_publication_needs_catalog_and_exact_webp(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        metadata.create_all(engine)
        file_uuid = "00000000-0000-0000-0000-000000000011"
        webp = f"/data/WRF/workbench/{TASK_ID}/rendered/d01/T2/frame.webp"
        with patch.object(source_resolver, "_ENGINE", engine):
            self.assertFalse(source_resolver.is_published_wrf_task(TASK_ID))
            with engine.begin() as connection:
                connection.execute(insert(public_info).values(
                    file_uuid=file_uuid, data_type="WRF", file_type="json",
                    original_file_name="scene.meta.json", stored_file_name="scene.meta.json",
                    source_path=f"workbench/{TASK_ID}/scene.meta.json",
                    file_size=1, file_hash="1" * 64, visibility="public",
                ))
                connection.execute(insert(wrf_info).values(
                    asset_uuid="00000000-0000-0000-0000-000000000012", file_uuid=file_uuid,
                    element_key="T2", webp_url=webp, asset_status="ready",
                ))
            self.assertTrue(source_resolver.is_published_wrf_task(TASK_ID))
            self.assertTrue(source_resolver.is_published_wrf_asset(webp))
            self.assertFalse(source_resolver.is_published_wrf_asset(webp.replace("frame", "other")))
            self.assertFalse(source_resolver.is_published_wrf_asset(
                f"/data/WRF/workbench/{TASK_ID}/task.request.json"))
        engine.dispose()

    def test_uploaded_wrf_catalog_enforces_owner_then_public_visibility(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        metadata.create_all(engine)
        file_uuid = "00000000-0000-0000-0000-000000000001"
        base = f"/data/WRF/assets/{file_uuid}/"
        with engine.begin() as connection:
            connection.execute(insert(public_info).values(
                file_uuid=file_uuid, user_uuid="alice", data_type="WRF",
                file_type="netcdf", original_file_name="wrfout", stored_file_name="wrfout",
                source_path="raw/wrfout", file_size=1, file_hash="0" * 64,
                meta_path=f"WRF/assets/{file_uuid}/scene.meta.json", visibility="private",
            ))
            connection.execute(insert(wrf_info).values(
                asset_uuid="00000000-0000-0000-0000-000000000002", file_uuid=file_uuid,
                element_key="T2", webp_url=base + "T2.webp", asset_status="ready",
            ))
        with patch.object(source_resolver, "_ENGINE", engine):
            visible = source_resolver.is_visible_wrf_upload_asset
            self.assertTrue(visible(base + "T2.webp", {"sub": "alice", "role": 1}))
            self.assertFalse(visible(base + "T2.webp", {"sub": "bob", "role": 1}))
            self.assertTrue(visible(base + "T2.webp", {"sub": "admin", "role": 3}))
            self.assertFalse(visible(base + "unlisted.webp", {"sub": "alice", "role": 1}))
            with engine.begin() as connection:
                connection.execute(public_info.update().where(public_info.c.file_uuid == file_uuid).values(visibility="public"))
            self.assertTrue(visible(base + "T2.webp", {"sub": "bob", "role": 1}))
        engine.dispose()

    def test_only_registered_workbench_assets_are_shared(self):
        base = f"/data/WRF/workbench/{TASK_ID}/"
        self.assertFalse(allow_wrf_asset(base + "scene.meta.json", lambda _: False, lambda _: True))
        self.assertTrue(allow_wrf_asset(base + "scene.meta.json", lambda _: True, lambda _: False))
        self.assertTrue(allow_wrf_asset(base + "rendered/d01/t2/frame.webp", lambda _: True, lambda _: True))
        self.assertFalse(allow_wrf_asset(base + "rendered/d01/t2/unlisted.webp", lambda _: True, lambda _: False))
        self.assertFalse(allow_wrf_asset(base + "task.request.json", lambda _: True, lambda _: True))
        self.assertFalse(allow_wrf_asset(base + "rendered/d01/t2/frame.nc", lambda _: True, lambda _: True))
        self.assertFalse(allow_wrf_asset(f"/data/WRF/runs/{TASK_ID}/scene.meta.json", lambda _: True, lambda _: True))
        self.assertFalse(allow_wrf_asset("/data/WRF/private/config.json", lambda _: True, lambda _: True))
        upload = "/data/WRF/assets/00000000-0000-0000-0000-000000000001/T2.webp"
        self.assertFalse(allow_wrf_asset(upload, lambda _: True, lambda _: True))
        self.assertTrue(allow_wrf_asset(upload, lambda _: False, lambda _: False,
                                        lambda path: path == upload))
        highres = "/data/WRF/workbench/ds_20260924T000000Z_1234abcd/"
        self.assertTrue(allow_wrf_asset(highres + "scene.meta.json", lambda _: True, lambda _: False))
        self.assertTrue(allow_wrf_asset(highres + "downscale.product.json", lambda _: True, lambda _: True))
        self.assertTrue(allow_wrf_asset(highres + "browser/wind.webp", lambda _: True, lambda _: True))
        self.assertFalse(allow_wrf_asset(highres + "browser/private.webp", lambda _: True, lambda _: False))


if __name__ == "__main__":
    unittest.main()
