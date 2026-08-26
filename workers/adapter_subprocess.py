from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from services.adapter_runner import run_adapter, run_collection_adapter


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exactly one weather Adapter job")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = Path(args.job).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_path = Path(job["result_path"]).resolve()
    error_path = Path(job["error_path"]).resolve()
    try:
        if job.get("task_kind") == "satellite_collection":
            result = run_collection_adapter(
                file_uuid=job["file_uuid"],
                collection_uuid=job["collection_uuid"],
                data_type=job["data_type"],
                members=job["members"],
                output_root=job["output_root"],
                attempt_dir=job["stage_dir"],
                input_dir=job["input_dir"],
            )
        else:
            result = run_adapter(
                file_uuid=job["file_uuid"],
                data_type=job["data_type"],
                source_path=job["source_path"],
                output_root=job["output_root"],
                attempt_dir=job["stage_dir"],
                collection_uuid=job.get("collection_uuid"),
                original_file_name=job.get("original_file_name"),
            )
        _write_json(result_path, result)
        return 0
    except Exception as exc:
        retryable = not isinstance(exc, (ValueError, ImportError, NotImplementedError, FileNotFoundError))
        _write_json(
            error_path,
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "retryable": retryable,
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
