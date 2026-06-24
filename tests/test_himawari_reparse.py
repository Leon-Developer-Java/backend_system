from pathlib import Path

from adapters.himawari_adapter import parse_hsd_filename, scan_hsd_scenes


def test_parse_hsd_filename_extracts_scene_fields():
    info = parse_hsd_filename("HS_H09_20260616_0000_B13_FLDK_R20_S0110.DAT.bz2")

    assert info == {
        "satellite": "Himawari-9",
        "date": "20260616",
        "time": "0000",
        "band": "B13",
        "region": "FLDK",
        "resolution": "R20",
        "segment": 1,
        "total_segments": 10,
    }


def test_scan_hsd_scenes_groups_complete_raw_directories(tmp_path):
    raw = tmp_path / "20260616" / "0000" / "raw"
    raw.mkdir(parents=True)
    for segment in range(1, 11):
        (raw / f"HS_H09_20260616_0000_B13_FLDK_R20_S{segment:02d}10.DAT.bz2").write_bytes(b"")
    ignored = tmp_path / "20260616" / "0010" / "raw"
    ignored.mkdir(parents=True)
    (ignored / "not-hsd.txt").write_text("x")

    scenes = scan_hsd_scenes(tmp_path, min_files=10)

    assert scenes == [
        {
            "date": "20260616",
            "time": "0000",
            "raw_dir": raw,
            "file_count": 10,
            "bands": ["B13"],
        }
    ]
