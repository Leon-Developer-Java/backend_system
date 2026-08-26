import os
import re
import sys
from pathlib import Path

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DB.config import WORKSPACE_ROOT, create_database_engine
from DB.schema import public_info


RAW_STORAGE_ROOT = Path(
    os.getenv("RAW_STORAGE_ROOT", str(WORKSPACE_ROOT / "storage" / "raw"))
).expanduser().resolve()
_FILE_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_ENGINE = create_database_engine()


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
