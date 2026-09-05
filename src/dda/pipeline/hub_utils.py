"""torch.hub preflight bypass; falls back to the local hub cache when GitHub is unreachable."""

import http.client
import logging
import urllib.error
from pathlib import Path

log = logging.getLogger(__name__)

_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    http.client.RemoteDisconnected,
    ConnectionError,
    TimeoutError,
)


def enable_offline_torch_hub_fallback() -> None:
    from torch import hub

    if getattr(hub, "_dda_offline_fallback_applied", False):
        return

    original_parse = hub._parse_repo_info

    def patched(github: str):
        try:
            return original_parse(github)
        except _NETWORK_ERRORS as err:
            owner_name = github.split(":", 1)[0]
            if "/" not in owner_name:
                raise
            owner, name = owner_name.split("/", 1)
            ref = github.split(":", 1)[1] if ":" in github else "main"
            cache_dir = Path(hub.get_dir()) / f"{owner}_{name}_{ref}"
            if not cache_dir.is_dir():
                raise
            log.warning(
                "torch.hub GitHub preflight failed (%s); falling back to local cache %s",
                err,
                cache_dir,
            )
            return owner, name, ref

    hub._parse_repo_info = patched  # ty: ignore[invalid-assignment]
    hub._dda_offline_fallback_applied = True  # ty: ignore[unresolved-attribute]
