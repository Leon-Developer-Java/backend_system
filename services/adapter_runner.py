from __future__ import annotations

import json
import importlib
import shutil
from pathlib import Path
from typing import Any

ADAPTERS = {
    "CMA": "adapters.cma_adapter",
    "ERA5": "adapters.era5_adapter",
    "GFS": "adapters.gfs_adapter",
    "ECMWF": "adapters.gfs_adapter",
    "FY3": "adapters.fy3_adapter",
    "RADAR": "adapters.radar_adapter",
    "WRF": "adapters.wrf_adapter",
}


def canonical_data_type(value: str) -> str:
    text = str(value or "").strip().upper().replace("-", "")
    if text == "RADAR" or text == "雷达":
        return "RADAR"
    if text in {"FY3", "FY"}:
        return "FY3"
    if text in ADAPTERS:
        return text
    raise ValueError(f"No Adapter Worker mapping for data_type={value!r}")


def _safe_source_name(value: str, fallback: str) -> str:
    name = Path(str(value or "")).name.strip()
    if not name or name in {".", ".."}:
        return fallback
    return name


def _select_meta_file(stage_dir: Path, meta: dict[str, Any]) -> Path:
    value = meta.get("meta_file") if isinstance(meta, dict) else None
    if value:
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = stage_dir / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(stage_dir.resolve())
        except ValueError:
            candidate = Path()
        if candidate.is_file():
            return candidate

    candidates = sorted(
        stage_dir.rglob("*.meta.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise ValueError("Adapter completed without a meta.json file")
    return candidates[0]


def run_adapter(
    file_uuid: str,
    data_type: str,
    source_path: str | Path,
    output_root: str | Path,
    attempt_dir: str | Path,
    collection_uuid: str | None = None,
    original_file_name: str | None = None,
) -> dict[str, Any]:
    data_type = canonical_data_type(data_type)
    module = importlib.import_module(ADAPTERS[data_type])
    source_path = Path(source_path).resolve()
    output_root = Path(output_root).resolve()
    stage_dir = Path(attempt_dir).resolve()
    _assert_within(stage_dir, output_root)
    if not source_path.is_file():
        raise FileNotFoundError(f"raw source does not exist: {source_path}")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=False)

    source_name = _safe_source_name(original_file_name, source_path.name)
    staged_source = stage_dir / source_name
    shutil.copy2(source_path, staged_source)

    adapter_data_type = "Radar" if data_type == "RADAR" else data_type
    process_options = {"sync_database": False} if data_type == "ERA5" else {}
    meta = module.process_file(
        str(staged_source),
        data_type=adapter_data_type,
        **process_options,
    )
    if not isinstance(meta, dict):
        raise ValueError("Adapter did not return a metadata object")
    meta_file = _select_meta_file(stage_dir, meta)
    webp_files = [path for path in stage_dir.rglob("*.webp") if path.is_file()]

    return {
        "file_uuid": file_uuid,
        "collection_uuid": collection_uuid,
        "data_type": data_type,
        "adapter_name": module.__name__.split(".")[-1],
        "adapter_version": str(getattr(module, "ADAPTER_VERSION", "1.0")),
        "stage_dir": stage_dir.as_posix(),
        "staged_source": staged_source.as_posix(),
        "meta_file": meta_file.as_posix(),
        "webp_count": len(webp_files),
    }


def _replace_paths(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, replacements) for item in value]
    if not isinstance(value, str):
        return value
    result = value
    for source, destination in replacements:
        result = result.replace(source, destination)
    return result


def _assert_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escaped configured product root: {path}") from exc


def publish_adapter_output(
    child_result: dict[str, Any],
    product_root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    product_root = product_root.resolve()
    data_type = canonical_data_type(child_result["data_type"])
    display_type = "Radar" if data_type == "RADAR" else data_type
    stage_dir = Path(child_result["stage_dir"]).resolve()
    type_root = (product_root / display_type).resolve()
    assets_root = (type_root / "assets").resolve()
    final_dir = (assets_root / child_result["file_uuid"]).resolve()
    _assert_within(stage_dir, type_root)
    _assert_within(final_dir, assets_root)

    staged_source = Path(child_result["staged_source"]).resolve()
    if staged_source.is_file():
        staged_source.unlink()

    stage_relative = stage_dir.relative_to(product_root).as_posix()
    final_relative = final_dir.relative_to(product_root).as_posix()
    replacements = [
        (stage_dir.as_posix(), final_dir.as_posix()),
        (str(stage_dir), str(final_dir)),
        (f"/data/{stage_relative}", f"/data/{final_relative}"),
        (f"data/{stage_relative}", f"data/{final_relative}"),
    ]

    # GFS falls back to /data/{basename} outside its normal data root. Map such
    # values to the actual published path without changing the meta structure.
    basename_map: dict[str, str | None] = {}
    for file_path in stage_dir.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(stage_dir).as_posix()
        published_url = f"/data/{final_relative}/{relative}"
        current = basename_map.get(file_path.name)
        basename_map[file_path.name] = published_url if current in {None, published_url} else None
    for basename, published_url in basename_map.items():
        if published_url:
            replacements.append((f"/data/{basename}", published_url))

    source_meta = Path(child_result["meta_file"]).resolve()
    _assert_within(source_meta, stage_dir)
    source_meta_relative = source_meta.relative_to(stage_dir)
    for json_file in stage_dir.rglob("*.json"):
        try:
            value = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        rewritten = _replace_paths(value, replacements)
        json_file.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8")

    if not list(stage_dir.rglob("*.webp")):
        raise ValueError("Adapter completed without a WebP display asset")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir.replace(final_dir)
    for empty_parent in (stage_dir.parent, stage_dir.parent.parent):
        try:
            empty_parent.rmdir()
        except OSError:
            break

    meta_file = final_dir / source_meta_relative
    if not meta_file.is_file():
        raise ValueError("Published meta.json is missing")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return meta, meta_file, final_dir


def cleanup_stage(stage_dir: str | Path, product_root: Path) -> None:
    candidate = Path(stage_dir).resolve()
    root = product_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return
    if candidate.is_dir() and ".adapter_staging" in candidate.parts:
        shutil.rmtree(candidate, ignore_errors=True)
