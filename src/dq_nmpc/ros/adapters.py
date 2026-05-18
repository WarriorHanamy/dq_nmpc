"""ROS2 message <-> Pydantic schema adapters.

All conversion between ROS msg types and internal schemas lives here,
keeping the math/nmpc layers free of ROS dependencies.
"""

from __future__ import annotations

from dq_nmpc.schema import ClassicalState, ControlCommand, ReferenceTrajectory, TrajectoryPoint

__all__ = [
    "odometry_to_classical",
    "classical_to_odometry",
    "position_cmd_to_trajectory",
    "wrench_from_control",
]


def odometry_to_classical(msg) -> ClassicalState:
    """Convert nav_msgs/Odometry to ClassicalState."""
    return ClassicalState(
        x=msg.pose.pose.position.x,
        y=msg.pose.pose.position.y,
        z=msg.pose.pose.position.z,
        vx=msg.twist.twist.linear.x,
        vy=msg.twist.twist.linear.y,
        vz=msg.twist.twist.linear.z,
        qw=msg.pose.pose.orientation.w,
        qx=msg.pose.pose.orientation.x,
        qy=msg.pose.pose.orientation.y,
        qz=msg.pose.pose.orientation.z,
        wx=msg.twist.twist.angular.x,
        wy=msg.twist.twist.angular.y,
        wz=msg.twist.twist.angular.z,
    )


def classical_to_odometry(state: ClassicalState, msg_type, stamp):
    """Convert ClassicalState to nav_msgs/Odometry message."""
    msg = msg_type()
    msg.header.stamp = stamp
    msg.header.frame_id = "world"
    msg.pose.pose.position.x = state.x
    msg.pose.pose.position.y = state.y
    msg.pose.pose.position.z = state.z
    msg.pose.pose.orientation.w = state.qw
    msg.pose.pose.orientation.x = state.qx
    msg.pose.pose.orientation.y = state.qy
    msg.pose.pose.orientation.z = state.qz
    msg.twist.twist.linear.x = state.vx
    msg.twist.twist.linear.y = state.vy
    msg.twist.twist.linear.z = state.vz
    msg.twist.twist.angular.x = state.wx
    msg.twist.twist.angular.y = state.wy
    msg.twist.twist.angular.z = state.wz
    return msg


def position_cmd_to_trajectory(msg) -> ReferenceTrajectory:
    """Convert quadrotor_msgs/PositionCommand to ReferenceTrajectory."""
    points = []
    for point in msg.points:
        tp = TrajectoryPoint(
            x=point.position.x,
            y=point.position.y,
            z=point.position.z,
            vx=point.velocity.x,
            vy=point.velocity.y,
            vz=point.velocity.z,
            qw=point.quaternion.w,
            qx=point.quaternion.x,
            qy=point.quaternion.y,
            qz=point.quaternion.z,
            wx=point.angular_velocity.x,
            wy=point.angular_velocity.y,
            wz=point.angular_velocity.z,
            thrust=point.force,
            torque_x=0.0,
            torque_y=0.0,
            torque_z=0.0,
        )
        points.append(tp)
    return ReferenceTrajectory(points=points, horizon_steps=len(points))


def wrench_from_control(control: ControlCommand, msg_type):
    """Convert ControlCommand to geometry_msgs/Wrench message."""
    msg = msg_type()
    msg.force.x = 0.0
    msg.force.y = 0.0
    msg.force.z = control.thrust
    msg.torque.x = control.torque_x
    msg.torque.y = control.torque_y
    msg.torque.z = control.torque_z
    return msg
