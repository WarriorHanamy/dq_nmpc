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


def visualize_trajectory(
    csv_path: Path,
    shape: str,
    inner_points: np.ndarray,
    sfc_centers: list[np.ndarray],
    half_extents: tuple[float, float, float],
    cost: float,
    output_path: Path,
    optimized_positions: np.ndarray | None = None,
) -> Path:
    """Generate interactive plotly HTML visualization of a trajectory.

    @param[in] csv_path             Path to trajectory CSV
    @param[in] shape                Trajectory shape name
    @param[in] inner_points         (3, N) seed waypoints
    @param[in] sfc_centers          List of (3,) SFC box centers
    @param[in] half_extents         SFC box half-extents [m]
    @param[in] cost                 Optimization cost
    @param[in] output_path          Output HTML path
    @param[in] optimized_positions  (3, M) GCOPTER-optimized junction positions
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
    w = data[:, _i("wx") : _i("wz") + 1]
    thrust = data[:, _i("thrust")]

    v_norm = np.linalg.norm(vel, axis=1)

    # Read torque from CSV if columns exist
    torque = None
    try:
        torque = data[:, _i("torque_x") : _i("torque_z") + 1]
    except ValueError:
        pass

    fig = make_subplots(
        rows=7,
        cols=2,
        column_widths=[0.55, 0.45],
        specs=[
            [{"type": "scene", "rowspan": 7}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=(
            "",
            "Position [m]",
            "Velocity & Body Rates",
            "Acceleration [m/s²]",
            "Jerk [m/s³]",
            "Snap [m/s⁴]",
            "Thrust [N]",
            "Torque [N·m]",
        ),
        vertical_spacing=0.04,
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
                marker=dict(color="magenta", size=6, symbol="circle-open"),
                name="optimized waypoints",
            ),
            row=1,
            col=1,
        )

    # SFC box wireframes
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
            marker=dict(color="green", size=8, symbol="circle"),
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

    # --- Row 2: velocity + body rates ---
    fig.add_trace(
        go.Scatter(x=t, y=v_norm, name="|v| [m/s]", line=dict(width=1.5)),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=w[:, 0],
            name="wx [rad/s]",
            line=dict(width=1, dash="dot"),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=w[:, 1],
            name="wy [rad/s]",
            line=dict(width=1, dash="dot"),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=w[:, 2],
            name="wz [rad/s]",
            line=dict(width=1, dash="dot"),
        ),
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

    # --- Row 6: thrust ---
    fig.add_trace(
        go.Scatter(x=t, y=thrust, name="thrust [N]", line=dict(width=1.5)),
        row=6,
        col=2,
    )

    # --- Row 7: torque ---
    if torque is not None:
        tau_labels = ["τx [N·m]", "τy [N·m]", "τz [N·m]"]
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=torque[:, i],
                    name=tau_labels[i],
                    line=dict(width=1.5),
                    legendgroup="torque",
                ),
                row=7,
                col=2,
            )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"GCOPTER Trajectory — {shape}  (cost={cost:.2f})",
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

    fig.update_xaxes(title_text="Time [s]", row=7, col=2)
    fig.update_yaxes(title_text="[m/s] / [rad/s]", row=2, col=2)
    fig.update_yaxes(title_text="[m/s²]", row=3, col=2)
    fig.update_yaxes(title_text="[m/s³]", row=4, col=2)
    fig.update_yaxes(title_text="[m/s⁴]", row=5, col=2)
    fig.update_yaxes(title_text="[N]", row=6, col=2)
    fig.update_yaxes(title_text="[N·m]", row=7, col=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Visualization saved: {output_path}")
    return output_path


def visualize_bullet_belt(
    belt,
    output_path: str | Path | None = None,
    *,
    step: int = 5,
) -> go.Figure:
    """Interactive 3D Plotly visualization of a bullet belt.

    Shows the full trajectory as a gray line with a slider-controlled
    highlighted segment (the current bullet).  Positions are extracted
    from the dual-quaternion dual part via ``position_from_dualquat_ca_func``.

    @param[in] belt         ``ReferenceTrajectoryAsBullets`` instance
    @param[in] output_path  Path for output ``.html`` (if None, figure is returned)
    @param[in] step         Show every N-th bullet as a frame (controls file size)
    @return                 Plotly ``Figure`` object
    """
    from dq_nmpc.math.dq_functions import position_from_dualquat_ca_func

    dq_to_pos = position_from_dualquat_ca_func()

    N_c = belt.N_c
    N = belt.horizon_steps

    dq_all = belt.bullets[:, :, :8].reshape(-1, 8).T  # (8, N_c * N)
    pos_all = np.array(dq_to_pos(dq_all)).T  # (N_c * N, 3)
    pos_reshaped = pos_all.reshape(N_c, N, 3)  # (N_c, N, 3)
    traj_pos = pos_reshaped[:, 0, :]  # (N_c, 3) — first point per bullet

    if belt.bullets.shape[0] < 2:
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
            title="Bullet Belt (empty — N_c < 2)",
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
    trace_bullet = go.Scatter3d(
        x=list(b0_pos[:, 0]),
        y=list(b0_pos[:, 1]),
        z=list(b0_pos[:, 2]),
        mode="lines+markers",
        marker=dict(size=3, color="crimson"),
        line=dict(color="crimson", width=4),
        name="current bullet",
    )

    fig = go.Figure(data=[trace_full, trace_bullet])

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
            f"Bullet Belt — {N_c} bullets × {N} horizon steps "
            f"(step={step}, {len(frame_indices)} frames)"
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
                currentvalue={"prefix": "bullet k="},
                pad=dict(t=40),
            )
        ],
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Bullet-belt visualization saved: {output_path}")

    return fig
