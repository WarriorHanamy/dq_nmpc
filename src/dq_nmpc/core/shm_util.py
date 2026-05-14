"""SHM (POSIX shared memory) helpers — wait, cleanup."""

from __future__ import annotations

import os
import time

from dq_nmpc.schema import SHMConfig


def _default_config() -> SHMConfig:
    return SHMConfig.default()


def wait_for_shm(timeout: float | None = None) -> None:
    """Block until both SHM segments exist.

    Raises RuntimeError if the segments do not appear within the timeout.

    @param[in] timeout  Maximum wait time [s]; defaults to SHMConfig.attach_timeout.
    """
    cfg = _default_config()
    if timeout is None:
        timeout = cfg.attach_timeout

    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(cfg.state_file) and os.path.exists(cfg.ctrl_file):
            return
        time.sleep(0.1)

    raise RuntimeError(f"SHM segments not created within {timeout:.1f}s")


def cleanup_shm() -> None:
    """Remove both SHM segment files.  No error if they are already absent."""
    cfg = _default_config()
    for f in (cfg.state_file, cfg.ctrl_file):
        try:
            os.unlink(f)
        except OSError:
            pass
