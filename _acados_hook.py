"""Build acados on demand before any acados imports."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

_logger = logging.getLogger(__name__)


def build_acados(project: Path) -> None:
    """Build acados into _acados_build/install/; idempotent."""

    acados_src = project / "deps" / "acados"
    acados_build = project / "_acados_build"
    acados_install = acados_build / "install"
    acados_lib = acados_install / "lib" / "libacados.so"
    acados_link = acados_src / "lib"

    if acados_lib.exists():
        return

    if acados_build.exists():
        shutil.rmtree(acados_build)

    acados_build.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "cmake",
            "-B",
            str(acados_build),
            "-S",
            str(acados_src),
            "-DACADOS_WITH_OSQP=ON",
            "-DACADOS_PYTHON=ON",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_C_FLAGS=-D_POSIX_C_SOURCE=200809L",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            f"-DCMAKE_INSTALL_PREFIX={acados_install}",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(acados_build), "-j"],
        check=True,
    )
    subprocess.run(
        ["cmake", "--install", str(acados_build), "--prefix", str(acados_install)],
        check=True,
    )

    link_libs = acados_link / "link_libs.json"
    if link_libs.exists():
        dest = acados_install / "lib" / "link_libs.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(link_libs, dest)

    if acados_link.is_symlink() or acados_link.is_file():
        acados_link.unlink()
    elif acados_link.is_dir():
        shutil.rmtree(acados_link)
    acados_link.symlink_to(str(acados_install / "lib"), target_is_directory=True)

    # acados codegen needs headers via ACADOS_SOURCE_DIR/include
    acados_inc = acados_src / "include"
    if acados_inc.is_symlink() or acados_inc.is_file():
        acados_inc.unlink()
    elif acados_inc.is_dir():
        shutil.rmtree(acados_inc)
    acados_inc.symlink_to(str(acados_install / "include"), target_is_directory=True)

    _logger.info("acados built and installed to %s", acados_install)
