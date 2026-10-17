"""Access policy for WRF files hosted by the display service."""

from collections.abc import Callable


def allow_wrf_asset(path: str, published: Callable[[str], bool], listed: Callable[[str], bool],
                    upload_visible: Callable[[str], bool] | None = None) -> bool:
    if not path.startswith("/data/WRF/"):
        return True
    parts = path.removeprefix("/data/WRF/").split("/")
    if parts[0] == "runs":
        return False
    if parts[0] == "assets":
        return bool(upload_visible and upload_visible(path))
    if parts[0] != "workbench":
        return False  # Unknown WRF paths are private until a catalog policy explicitly lists them.
    if len(parts) < 3 or any(part in {"", ".", ".."} for part in parts):
        return False
    asset = parts[2:]
    if asset == ["scene.meta.json"]:
        return published(parts[1])
    if parts[1].startswith("ds_"):
        return listed(path)
    if not (
        len(asset) == 4 and asset[0] == "rendered"
        and asset[1] in {"d01", "d02", "d03", "d04"}
        and asset[3].endswith(".webp")
    ):
        return False
    return listed(path)
