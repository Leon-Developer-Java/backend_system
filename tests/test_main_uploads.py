from io import BytesIO
import sys
from types import SimpleNamespace


sys.modules.setdefault("cfgrib", SimpleNamespace(open_datasets=lambda *args, **kwargs: []))
sys.modules.setdefault("rasterio", SimpleNamespace())

import main


class FakeUpload:
    def __init__(self, filename: str, payload: bytes = b"data") -> None:
        self.filename = filename
        self.file = BytesIO(payload)


def test_infer_business_type_recognizes_himawari_hsd_bz2_before_generic_bz2():
    assert main.infer_business_type("HS_H09_20260616_0000_B13_FLDK_R20_S0110.DAT.bz2") == "Himawari"


def test_himawari_uploads_are_grouped_into_scene_raw_directory(tmp_path):
    files = [
        FakeUpload("HS_H09_20260616_0000_B13_FLDK_R20_S0110.DAT.bz2", b"one"),
        FakeUpload("HS_H09_20260616_0000_B13_FLDK_R20_S0210.DAT.bz2", b"two"),
    ]

    saved = main.save_upload_files(files, tmp_path / "Himawari", "Himawari")

    assert saved == [
        tmp_path / "Himawari" / "20260616" / "0000" / "raw" / files[0].filename,
        tmp_path / "Himawari" / "20260616" / "0000" / "raw" / files[1].filename,
    ]
    assert saved[0].read_bytes() == b"one"
    assert saved[1].read_bytes() == b"two"


def test_himawari_directory_upload_ignores_unrelated_files_for_type_and_save():
    files = [
        FakeUpload(".DS_Store", b"noise"),
        FakeUpload("HS_H09_20260616_0000_B13_FLDK_R20_S0110.DAT.bz2", b"hsd"),
    ]

    business_type = main.infer_upload_business_type(files)
    filtered = main.ADAPTERS[business_type].select_upload_files(files)

    assert business_type == "Himawari"
    assert [item.filename for item in filtered] == ["HS_H09_20260616_0000_B13_FLDK_R20_S0110.DAT.bz2"]
