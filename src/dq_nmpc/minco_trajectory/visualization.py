"""Plotly-based quadrotor trajectory visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dq_nmpc.schema import csv_column_index


def _box_wireframe(center: np.ndarray, half_extents: tuple[float, float, float]) -> np.ndarray:
    """Return (3, N) array of box edge vertices with NaN separators."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    corners = np.array(
        [
            [cx - hx, cy - hy, cz - hz],  # 0: ---
            [cx + hx, cy - hy, cz - hz],  # 1: +--
            [cx - hx, cy + hy, cz - hz],  # 2: -+-
            [cx + hx, cy + hy, cz - hz],  # 3: ++-
            [cx - hx, cy - hy, cz + hz],  # 4: --+
            [cx + hx, cy - hy, cz + hz],  # 5: +-+
            [cx - hx, cy + hy, cz + hz],  # 6: -++
            [cx + hx, cy + hy, cz + hz],  # 7: +++
        ]
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    nan = np.array([np.nan, np.nan, np.nan])
    segs = [np.stack([corners[i], corners[j], nan]) for i, j in edges]
    return np.vstack(segs).T


def _quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw,qx,qy,qz] to ZYX Euler angles in degrees.

    ZYX intrinsic rotation = Rz(yaw) * Ry(pitch) * Rx(roll).

    @param[in] q  (4,) or (N,4) quaternion array [qw, qx, qy, qz]
    @return       (3,) or (N,3) Euler angles [roll, pitch, yaw] in degrees
    """
    q = np.atleast_2d(q)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R00 = 1.0 - 2.0 * (qy**2 + qz**2)
    R10 = 2.0 * (qx * qy + qz * qw)
    R20 = 2.0 * (qx * qz - qy * qw)
    R21 = 2.0 * (qy * qz + qx * qw)
    R22 = 1.0 - 2.0 * (qx**2 + qy**2)

    roll = np.arctan2(R21, R22)
    pitch = -np.arcsin(np.clip(R20, -1.0, 1.0))
    yaw = np.arctan2(R10, R00)

    euler = np.column_stack([np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)])
    if euler.shape[0] == 1 and q.shape[0] == 1:
        return euler[0]
    return euler


def _make_title(shape: str, cost: float) -> str:
    """Build title string from optional shape and cost."""
    parts = ["GCOPTER Trajectory"]
    if shape:
        parts.append(f"— {shape}")
    if cost > 0.0:
        parts.append(f"(cost={cost:.2f})")
    return " ".join(parts)


def visualize_trajectory(
    csv_path: Path,
    output_path: Path,
    *,
    shape: str = "",
    cost: float = 0.0,
    inner_points: np.ndarray | None = None,
    sfc_centers: list[np.ndarray] | None = None,
    half_extents: tuple[float, float, float] | None = None,
    optimized_positions: np.ndarray | None = None,
) -> Path:
    """Generate interactive plotly HTML visualization of a trajectory.

    @param[in] csv_path             Path to trajectory CSV
    @param[in] output_path          Output HTML path
    @param[in] shape                Trajectory shape name (title only)
    @param[in] cost                 Optimization cost (title only)
    @param[in] inner_points         (3, N) seed waypoints (optional)
    @param[in] sfc_centers          List of (3,) SFC box centers (optional)
    @param[in] half_extents         SFC box half-extents [m] (optional)
    @param[in] optimized_positions  (3, M) GCOPTER-optimized junction positions (optional)
    @return                         Path to the written HTML file
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    _i = csv_column_index
    t = data[:, _i("t")]
    pos = data[:, _i("x") : _i("z") + 1]
    vel = data[:, _i("vx") : _i("vz") + 1]
    acc = data[:, _i("ax") : _i("az") + 1]
    jer = data[:, _i("jx") : _i("jz") + 1]
    sna = data[:, _i("sx") : _i("sz") + 1]

    v_norm = np.linalg.norm(vel, axis=1)

    fig = make_subplots(
        rows=5,
        cols=2,
        column_widths=[0.55, 0.45],
        specs=[
            [{"type": "scene", "rowspan": 5}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=(
            "",
            "Position [m]",
            "Velocity [m/s]",
            "Acceleration [m/s²]",
            "Jerk [m/s³]",
            "Snap [m/s⁴]",
        ),
        vertical_spacing=0.05,
    )

    # --- Left panel: 3D trajectory ---
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines",
            line=dict(color="royalblue", width=3),
            name="trajectory",
        ),
        row=1,
        col=1,
    )

    # Seed waypoints
    if inner_points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=inner_points[0],
                y=inner_points[1],
                z=inner_points[2],
                mode="markers",
                marker=dict(color="orange", size=4, symbol="diamond"),
                name="seed waypoints",
            ),
            row=1,
            col=1,
        )

    # Optimized waypoints (GCOPTER junction positions)
    if optimized_positions is not None:
        fig.add_trace(
            go.Scatter3d(
                x=optimized_positions[0],
                y=optimized_positions[1],
                z=optimized_positions[2],
                mode="markers",
                marker=dict(color="green", size=8, symbol="diamond"),
                name="optimized waypoints",
            ),
            row=1,
            col=1,
        )

    # SFC box wireframes
    if sfc_centers is not None and half_extents is not None:
        for i, center in enumerate(sfc_centers):
            wire = _box_wireframe(center, half_extents)
            fig.add_trace(
                go.Scatter3d(
                    x=wire[0],
                    y=wire[1],
                    z=wire[2],
                    mode="lines",
                    line=dict(color="dimgray", width=2),
                    opacity=0.35,
                    showlegend=(i == 0),
                    name="SFC",
                ),
                row=1,
                col=1,
            )

    # Start / end markers
    fig.add_trace(
        go.Scatter3d(
            x=[pos[0, 0]],
            y=[pos[0, 1]],
            z=[pos[0, 2]],
            mode="markers",
            marker=dict(color="green", size=10, symbol="diamond"),
            name="start",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[pos[-1, 0]],
            y=[pos[-1, 1]],
            z=[pos[-1, 2]],
            mode="markers",
            marker=dict(color="red", size=8, symbol="circle"),
            name="end",
        ),
        row=1,
        col=1,
    )

    # --- Row 1: position ---
    fig.add_trace(
        go.Scatter(x=t, y=pos[:, 0], name="x", line=dict(width=1.5), legendgroup="pos"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=pos[:, 1], name="y", line=dict(width=1.5), legendgroup="pos"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=pos[:, 2], name="z", line=dict(width=1.5), legendgroup="pos"),
        row=1,
        col=2,
    )

    # --- Row 2: speed ---
    fig.add_trace(
        go.Scatter(x=t, y=v_norm, name="|v| [m/s]", line=dict(width=1.5)),
        row=2,
        col=2,
    )

    # --- Row 3: acceleration ---
    fig.add_trace(
        go.Scatter(x=t, y=acc[:, 0], name="ax", line=dict(width=1.5), legendgroup="acc"),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=acc[:, 1], name="ay", line=dict(width=1.5), legendgroup="acc"),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=acc[:, 2], name="az", line=dict(width=1.5), legendgroup="acc"),
        row=3,
        col=2,
    )

    # --- Row 4: jerk ---
    fig.add_trace(
        go.Scatter(x=t, y=jer[:, 0], name="jx", line=dict(width=1.5), legendgroup="jer"),
        row=4,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=jer[:, 1], name="jy", line=dict(width=1.5), legendgroup="jer"),
        row=4,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=jer[:, 2], name="jz", line=dict(width=1.5), legendgroup="jer"),
        row=4,
        col=2,
    )

    # --- Row 5: snap ---
    fig.add_trace(
        go.Scatter(x=t, y=sna[:, 0], name="sx", line=dict(width=1.5), legendgroup="sna"),
        row=5,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sna[:, 1], name="sy", line=dict(width=1.5), legendgroup="sna"),
        row=5,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sna[:, 2], name="sz", line=dict(width=1.5), legendgroup="sna"),
        row=5,
        col=2,
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=_make_title(shape, cost),
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    fig.update_xaxes(title_text="Time [s]", row=5, col=2)
    fig.update_yaxes(title_text="[m/s]", row=2, col=2)
    fig.update_yaxes(title_text="[m/s²]", row=3, col=2)
    fig.update_yaxes(title_text="[m/s³]", row=4, col=2)
    fig.update_yaxes(title_text="[m/s⁴]", row=5, col=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Visualization saved: {output_path}")
    return output_path


def visualize_belt_set(
    belts,
    output_path: str | Path | None = None,
    *,
    step: int = 5,
) -> go.Figure:
    """Interactive 3D Plotly visualization of a belt set.

    Shows the full trajectory as a gray line with a slider-controlled
    highlighted segment (the current belt).  Positions are extracted
    from the dual-quaternion dual part via ``position_from_dualquat_ca_func``.

    @param[in] belts       ``RefTrajectoryAsBelts`` instance
    @param[in] output_path  Path for output ``.html`` (if None, figure is returned)
    @param[in] step         Show every N-th belt as a frame (controls file size)
    @return                 Plotly ``Figure`` object
    """
    from dq_nmpc.math.dq_functions import position_from_dualquat_ca_func

    dq_to_pos = position_from_dualquat_ca_func()

    N_c = belts.N_c
    N = belts.horizon_steps

    dq_all = belts.belts[:, :, :8].reshape(-1, 8).T  # (8, N_c * N)
    pos_all = np.array(dq_to_pos(dq_all)).T  # (N_c * N, 3)
    pos_reshaped = pos_all.reshape(N_c, N, 3)  # (N_c, N, 3)
    traj_pos = pos_reshaped[:, 0, :]  # (N_c, 3) — first point per belt

    if belts.belts.shape[0] < 2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=list(traj_pos[:, 0]),
                y=list(traj_pos[:, 1]),
                z=list(traj_pos[:, 2]),
                mode="markers+lines",
                marker=dict(size=3),
                line=dict(color="gray", width=1),
                name="trajectory",
            )
        )
        fig.update_layout(
            title="Belt Set (empty — N_c < 2)",
            scene=dict(aspectmode="data"),
        )
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
        return fig

    trace_full = go.Scatter3d(
        x=list(traj_pos[:, 0]),
        y=list(traj_pos[:, 1]),
        z=list(traj_pos[:, 2]),
        mode="lines",
        line=dict(color="lightgray", width=1),
        name="full trajectory",
    )

    b0_pos = pos_reshaped[0]  # (N, 3)
    trace_belt = go.Scatter3d(
        x=list(b0_pos[:, 0]),
        y=list(b0_pos[:, 1]),
        z=list(b0_pos[:, 2]),
        mode="lines+markers",
        marker=dict(size=3, color="crimson"),
        line=dict(color="crimson", width=4),
        name="current belt",
    )

    fig = go.Figure(data=[trace_full, trace_belt])

    frames: list[go.Frame] = []
    slider_steps: list[dict] = []
    frame_indices = list(range(0, N_c, step))
    for k in frame_indices:
        pos_k = pos_reshaped[k]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=list(traj_pos[:, 0]),
                        y=list(traj_pos[:, 1]),
                        z=list(traj_pos[:, 2]),
                        mode="lines",
                        line=dict(color="lightgray", width=1),
                    ),
                    go.Scatter3d(
                        x=list(pos_k[:, 0]),
                        y=list(pos_k[:, 1]),
                        z=list(pos_k[:, 2]),
                        mode="lines+markers",
                        marker=dict(size=3, color="crimson"),
                        line=dict(color="crimson", width=4),
                    ),
                ],
                name=f"k={k}",
            )
        )
        slider_steps.append(
            dict(
                args=[
                    [f"k={k}"],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                ],
                label=str(k),
                method="animate",
            )
        )

    fig.frames = frames

    fig.update_layout(
        title=(
            f"Belt Set — {N_c} belts × {N} horizon steps (step={step}, {len(frame_indices)} frames)"
        ),
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                currentvalue={"prefix": "belt k="},
                pad=dict(t=40),
            )
        ],
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Belt-set visualization saved: {output_path}")

    return fig


def visualize_belt_euler_zyx(
    belts,
    dt: float,
    output_path: str | Path | None = None,
    *,
    step: int = 5,
) -> go.Figure:
    """Interactive Plotly visualization of ZYX Euler angles over each belt.

    For each belt k, extracts the quaternion from ``belts[k, :, 0:4]``,
    converts to ZYX Euler angles in degrees, and plots roll / pitch / yaw
    against horizon time (index * dt).  A slider scrubs through all N_c belts.
    The x-axis starts at 0 = belt start point.

    @param[in] belts       ``RefTrajectoryAsBelts`` instance
    @param[in] dt          Horizon time step [s]
    @param[in] output_path  Path for output ``.html`` (if None, figure is returned)
    @param[in] step         Show every N-th belt as a frame (controls file size)
    @return                 Plotly ``Figure`` object
    """
    N_c = belts.N_c
    N = belts.horizon_steps
    horizon_time = np.arange(N) * dt

    quat_all = belts.belts[:, :, :4].reshape(-1, 4)
    euler_all = _quat_to_euler_zyx(quat_all)
    euler_reshaped = euler_all.reshape(N_c, N, 3)

    labels = ["roll", "pitch", "yaw"]
    colors = ["crimson", "royalblue", "darkgreen"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Roll [deg]", "Pitch [deg]", "Yaw [deg]"),
        vertical_spacing=0.06,
    )

    e0 = euler_reshaped[0]
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=horizon_time,
                y=e0[:, i],
                mode="lines+markers",
                marker=dict(size=2, color=colors[i]),
                line=dict(width=2, color=colors[i]),
                name=labels[i],
            ),
            row=i + 1,
            col=1,
        )

    frames: list[go.Frame] = []
    slider_steps: list[dict] = []
    frame_indices = list(range(0, N_c, step))
    for k in frame_indices:
        ek = euler_reshaped[k]
        frame_data = [
            go.Scatter(
                x=horizon_time,
                y=ek[:, i],
                mode="lines+markers",
                marker=dict(size=2, color=colors[i]),
                line=dict(width=2, color=colors[i]),
            )
            for i in range(3)
        ]
        frames.append(go.Frame(data=frame_data, name=f"k={k}"))
        slider_steps.append(
            dict(
                args=[
                    [f"k={k}"],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                ],
                label=str(k),
                method="animate",
            )
        )

    fig.frames = frames

    fig.update_layout(
        title=(
            f"Belt Euler ZYX — {N_c} belts × {N} horizon steps"
            f"  (step={step}, {len(frame_indices)} frames, dt={dt:.3f} s)"
        ),
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                currentvalue={"prefix": "belt k="},
                pad=dict(t=50),
            )
        ],
    )

    fig.update_xaxes(title_text="Horizon time [s]", row=3, col=1)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Belt Euler ZYX visualization saved: {output_path}")

    return fig
