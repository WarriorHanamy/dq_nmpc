"""Static NMPC reference trajectory visualization pipeline.

Load minco NPZ → reinterpret via NMPC flatness → write Rerun .rrd.
"""

from __future__ import annotations

from pathlib import Path

from dq_nmpc.minco_trajectory import load_trajectory_npz
from dq_nmpc.nmpc import dense_ref_from_minco, visualize_ref_params
from dq_nmpc.schema import NMPCConfig


def visualize_reference(
    npz_path: str | Path,
    config_path: str | Path,
    output_path: str | Path | None = None,
    *,
    spawn: bool = False,
) -> Path:
    """Visualize an NMPC reinterpreted reference trajectory as a static Rerun .rrd.

    @param[in] npz_path     Path to minco trajectory.npz
    @param[in] config_path  Path to nmpc.yaml
    @param[in] output_path  Output .rrd path (default: derived from npz)
    @param[in] spawn        Spawn Rerun viewer after writing
    @return                 Path to the written .rrd file
    """
    npz_path = Path(npz_path)
    config_path = Path(config_path)

    if output_path is None:
        output_path = npz_path.with_suffix(".rrd")
    else:
        output_path = Path(output_path)

    traj7 = load_trajectory_npz(npz_path)
    config = NMPCConfig.from_yaml(config_path)

    ref_params = dense_ref_from_minco(traj7, config)
    dt = config.ocp.control_update_interval

    return visualize_ref_params(ref_params, output_path, dt=dt, spawn=spawn)
