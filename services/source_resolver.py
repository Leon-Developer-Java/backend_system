import os
import json
import re
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DB.config import WORKSPACE_ROOT, create_database_engine
from DB.schema import public_info, wrf_info


RAW_STORAGE_ROOT = Path(
    os.getenv("RAW_STORAGE_ROOT", str(WORKSPACE_ROOT / "storage" / "raw"))
).expanduser().resolve()
_FILE_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_ENGINE = create_database_engine()


def is_published_wrf_task(task_id: str) -> bool:
    """Use the catalog registration, not a file's existence, as publication proof."""
    if not re.fullmatch(r"(?:wrf_(?:gfs|ecmwf)|ds)_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}", task_id):
        return False
    try:
        with _ENGINE.connect() as connection:
            found = connection.execute(
                select(public_info.c.file_uuid).where(
                    public_info.c.data_type == "WRF",
                    public_info.c.source_path == f"workbench/{task_id}/scene.meta.json",
                    public_info.c.is_deleted.is_(False),
                ),
            ).first()
        return found is not None
    except SQLAlchemyError:
        return False


def is_published_wrf_asset(path: str) -> bool:
    """Only a URL explicitly listed in the registered WRF catalog is shared."""
    match = re.fullmatch(r"/data/WRF/workbench/((?:wrf_(?:gfs|ecmwf)|ds)_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8})/(.+)", path)
    if match is None:
        return False
    task_id, relative = match.groups()
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        return False
    if task_id.startswith("ds_"):
        if not is_published_wrf_task(task_id):
            return False
        if relative == "downscale.product.json":
            return True
        manifest_path = Path(__file__).resolve().parents[1] / "data" / "WRF" / "workbench" / task_id / "downscale.product.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assets = manifest.get("assets") if manifest.get("workflow_id") == task_id else None
            return isinstance(assets, list) and any(
                isinstance(asset, dict) and asset.get("path") == relative for asset in assets
            )
        except (OSError, ValueError, TypeError):
            return False
    if not re.fullmatch(r"rendered/d0[1-4]/[^/]+/[^/]+\.webp", relative):
        return False
    try:
        with _ENGINE.connect() as connection:
            found = connection.execute(
                select(wrf_info.c.asset_uuid).select_from(
                    wrf_info.join(public_info, wrf_info.c.file_uuid == public_info.c.file_uuid)
                ).where(
                    public_info.c.data_type == "WRF",
                    public_info.c.source_path == f"workbench/{task_id}/scene.meta.json",
                    public_info.c.is_deleted.is_(False),
                    wrf_info.c.webp_url == path,
                    wrf_info.c.asset_status == "ready",
                ),
            ).first()
        return found is not None
    except SQLAlchemyError:
        return False


def is_visible_wrf_upload_asset(path: str, user: dict) -> bool:
    """Keep parsed WRF uploads owner-only until their catalog visibility is public."""
    match = re.fullmatch(
        r"/data/WRF/assets/([0-9a-fA-F-]{36})/(.+)", path,
    )
    if match is None or not _FILE_UUID_PATTERN.fullmatch(match.group(1)):
        return False
    if any(part in {"", ".", ".."} for part in match.group(2).split("/")):
        return False
    file_uuid = match.group(1)
    try:
        with _ENGINE.connect() as connection:
            catalog = connection.execute(select(public_info).where(
                public_info.c.file_uuid == file_uuid,
                public_info.c.data_type == "WRF",
                public_info.c.is_deleted.is_(False),
            )).mappings().first()
            if catalog is None:
                return False
            owner = str(catalog.get("user_uuid") or "")
            if catalog.get("visibility") != "public" and not (
                int(user.get("role") or 0) >= 3 or (owner and owner == user.get("sub"))
            ):
                return False
            if path == f"/data/{catalog.get('meta_path')}":
                return True
            asset = connection.execute(select(wrf_info.c.asset_uuid).where(
                wrf_info.c.file_uuid == file_uuid,
                wrf_info.c.webp_url == path,
                wrf_info.c.asset_status == "ready",
            )).first()
            return asset is not None
    except SQLAlchemyError:
        return False


def resolve_raw_source(path_hint: str | Path | None, data_type: str | None = None) -> Path | None:
    """Resolve a published asset path back to its private raw source file."""
    if not path_hint:
        return None

    hint = Path(path_hint).expanduser()
    if hint.is_file():
        return hint.resolve()

    file_uuid = next((part for part in reversed(hint.parts) if _FILE_UUID_PATTERN.fullmatch(part)), None)
    if not file_uuid:
        return None

    conditions = [
        public_info.c.file_uuid == file_uuid,
        public_info.c.is_deleted.is_(False),
    ]
    if data_type:
        conditions.append(public_info.c.data_type == data_type.upper())

    with _ENGINE.connect() as connection:
        source_path = connection.execute(
            select(public_info.c.source_path).where(*conditions)
        ).scalar_one_or_none()
    if not source_path:
        return None

    candidate = (RAW_STORAGE_ROOT / str(source_path)).resolve()
    try:
        candidate.relative_to(RAW_STORAGE_ROOT)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None
