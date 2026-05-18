#!/usr/bin/env python3
import time

import casadi as ca
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Wrench
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker

from dq_nmpc.nmpc.controller import solver
from dq_nmpc.nmpc.dynamics import (
    dual_velocity_casadi,
    dualquat_quat_casadi,
    dualquat_trans_casadi,
    error_dual_aux_casadi,
    rotation_casadi,
    rotation_inverse_casadi,
    velocities_from_twist_casadi,
)

# Libraries of dual-quaternions
from dq_nmpc.nmpc.functions import dualquat_from_pose_casadi
from dq_nmpc.ros.adapters import (
    odometry_to_classical,
    position_cmd_to_trajectory,
    wrench_from_control,
)
from dq_nmpc.schema import ControlCommand, NMPCConfig

# Function to create a dualquaternion, get quaernion and translatation and returns a dualquaternion
dualquat_from_pose = dualquat_from_pose_casadi()

# Function to get the trasnlation from the dualquaternion, input dualquaternion and get a translation expressed as a quaternion [0.0, tx, ty,tz]
get_trans = dualquat_trans_casadi()

# Function to get the quaternion from the dualquaternion, input dualquaternion and get a the orientation quaternions [qw, qx, qy, qz]
get_quat = dualquat_quat_casadi()

# Function that maps linear velocities in the inertial frame and angular velocities in the body frame to both of them in the body frame, this is known as twist using dualquaternions
dual_twist = dual_velocity_casadi()

# Function that maps linear and angular velocites in the body frame to the linear velocity in the inertial frame and the angular velocity still in th body frame
velocity_from_twist = velocities_from_twist_casadi()

# Function that returns a vector from the body frame to the inertial frame
rot = rotation_casadi()

# Function that returns a vector from the inertial frame to the body frame
inverse_rot = rotation_inverse_casadi()

# Function to check for the shorthest path
error_dual_f = error_dual_aux_casadi()


class DQnmpcNode(Node):
    def __init__(self):
        super().__init__("DQNMPC_FINAL")
        # Lets define internal variables
        self.declare_parameter("mass", 1.0)
        self.declare_parameter("gravity", 9.8)
        self.declare_parameter("ixx", 0.00305587)
        self.declare_parameter("iyy", 0.00159695)
        self.declare_parameter("izz", 0.00159687)
        self.declare_parameter("mav_name", "quadrotor")

        self.declare_parameter("nmpc_config_path", "")
        self.declare_parameter("flag_build", True)

        self.flag_build = self.get_parameter("flag_build").value
        self.mav_name = self.get_parameter("mav_name").get_parameter_value().string_value

        # Load NMPC solver config from standalone YAML
        nmpc_config_path = self.get_parameter("nmpc_config_path").get_parameter_value().string_value
        if not nmpc_config_path:
            self.get_logger().error("nmpc_config_path not set, cannot load NMPC config")
            raise RuntimeError("nmpc_config_path parameter is required")
        nmpc_config = NMPCConfig.from_yaml(nmpc_config_path)
        params = nmpc_config.to_params_dict()

        # Values of the system
        self.g = params["gravity"]
        self.mQ = params["mass"]

        # Inertia Matrix
        self.Jxx = params["ixx"]
        self.Jyy = params["iyy"]
        self.Jzz = params["izz"]
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])
        self.L = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g]

        # Initial States dual set zeros
        # Position of the system
        pos_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Linear velocity of the sytem respect to the inertial frame
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Angular velocity respect to the Body frame
        omega_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Initial Orientation expressed as quaternionn
        quat_0 = np.array([1.0, 0.0, 0.0, 0.0])

        # Auxiliary vector [x, v, q, w], which is used to update the odometry and the states of the system
        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0))

        ## Compute desired path based on read Odometry
        self.Q = np.array(params["nmpc"]["Q"])
        self.Q_e = np.array(params["nmpc"]["Q_e"])
        self.R = np.array(params["nmpc"]["R"])

        self.acados_ocp_solver, self.ocp = solver(params, self.flag_build)

        # Define odometry subscriber for the drone
        self.subscriber_ = self.create_subscription(
            Odometry, "odom", self.callback_get_odometry, 10
        )

        # Define planner subscriber for the drone
        self.subscriber_planner_ = self.create_subscription(
            PositionCommand, "position_cmd", self.callback_get_planner, 10
        )

        # Define odometry publisher for the desired path
        self.ref_msg = Odometry()
        self.publisher_ref_ = self.create_publisher(Odometry, "desired_frame", 10)

        # Definition of the publihser for the desired parth
        self.marker_msg = Marker()
        self.points = None
        self.publisher_ref_trajectory_ = self.create_publisher(Marker, "desired_path", 10)

        # Definition of the publisher
        self.control_msg = Wrench()
        self.publisher_control_ = self.create_publisher(Wrench, "cmd", 10)

        # Definition of the prediction time in secs
        self.t_N = params["nmpc"]["horizon_time"]

        # Definition of the horizon
        self.N_prediction = params["nmpc"]["horizon_steps"]

        # Sample time
        self.ts = params["nmpc"]["ts"]

        # Init states formulated as dualquaternions
        self.dual_1 = dualquat_from_pose(
            self.x_0[6],
            self.x_0[7],
            self.x_0[8],
            self.x_0[9],
            self.x_0[0],
            self.x_0[1],
            self.x_0[2],
        )

        # Init linear velocity in the inertial frame and angular velocity in the body frame
        self.angular_linear_1 = np.array(
            [self.x_0[10], self.x_0[11], self.x_0[12], self.x_0[3], self.x_0[4], self.x_0[5]]
        )  # Angular Body linear Inertial

        # Init Dual Twist
        self.dual_twist_1 = dual_twist(self.angular_linear_1, self.dual_1)

        # Auxiliar vector where we can to save all the information formulated as dualquaternion
        self.X = np.zeros((14, 1), dtype=np.double)
        self.X[:, 0] = np.array(ca.vertcat(self.dual_1, self.dual_twist_1)).reshape((14,))

        ## Auxiliar variables for the controller
        self.dual_1_control = dualquat_from_pose(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.angular_linear_1_control = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # Angular Body linear Inertial
        self.dual_twist_1_control = dual_twist(self.angular_linear_1_control, self.dual_1_control)
        self.X_control = np.zeros((14, 1), dtype=np.double)
        self.X_control[:, 0] = np.array(
            ca.vertcat(self.dual_1_control, self.dual_twist_1_control)
        ).reshape((14,))
        self.u_control = np.zeros((4, 1), dtype=np.double)
        self.u_control[0, 0] = self.g * self.mQ

        # Reference signals of the nmpc
        self.x_ref = np.zeros((13, self.N_prediction), dtype=np.double)
        self.u_d = np.zeros((4, self.N_prediction), dtype=np.double)
        self.w_dot_ref = np.zeros((3, self.N_prediction), dtype=np.double)
        self.X_d = np.zeros((14, self.N_prediction), dtype=np.double)

        self.init_marker()

        self.timer = self.create_timer(self.ts, self.control_nmpc)  # 0.01 seconds = 100 Hz
        self.start_time = time.time()

    def callback_get_planner(self, msg):
        # Parse trajectory via validated schema
        trajectory = position_cmd_to_trajectory(msg)

        for i, tp in enumerate(trajectory.points):
            if i >= self.N_prediction:
                break

            # Desired States
            self.x_ref[0:3, i] = np.array([tp.x, tp.y, tp.z])
            self.x_ref[3:6, i] = np.array([tp.vx, tp.vy, tp.vz])
            self.x_ref[6:10, i] = np.array([tp.qw, tp.qx, tp.qy, tp.qz])
            self.x_ref[10:13, i] = np.array([tp.wx, tp.wy, tp.wz])

            # Desired dualquaternion
            dual_1_d = dualquat_from_pose(tp.qw, tp.qx, tp.qy, tp.qz, tp.x, tp.y, tp.z)

            # Linear velocities inertial frame + angular velocities body frame
            angular_linear_1_d = np.array([tp.wx, tp.wy, tp.wz, tp.vx, tp.vy, tp.vz])
            # Init Dual Twist
            dual_twist_1_d = dual_twist(angular_linear_1_d, dual_1_d)

            # Update Reference
            self.X_d[8:14, i] = np.array(dual_twist_1_d).reshape((6,))
            self.X_d[0:8, i] = np.array(dual_1_d).reshape((8,))

            # Desired force
            self.u_d[0, i] = tp.thrust

            # Desired Torques
            self.w_dot_ref[0:3, i] = np.array([tp.torque_x, tp.torque_y, tp.torque_z])
            self.u_d[1:4, i] = self.J @ self.w_dot_ref[0:3, i] + np.cross(
                self.x_ref[10:13, i], self.J @ self.x_ref[10:13, i]
            )

        # Send data
        self.send_marker()
        self.send_ref()
        return None

    def callback_get_odometry(self, msg):
        # Parse odometry via validated schema
        state = odometry_to_classical(msg)

        # Rotation: convert body-frame linear velocity to inertial frame
        rotational = R.from_quat([state.qx, state.qy, state.qz, state.qw])
        rotational_matrix = rotational.as_matrix()
        vb = np.array(
            [[msg.twist.twist.linear.x], [msg.twist.twist.linear.y], [msg.twist.twist.linear.z]]
        )
        vx_i = rotational_matrix @ vb

        # Build 13D classical state array (backward-compatible)
        self.x_0 = np.array(
            [
                state.x,
                state.y,
                state.z,
                vx_i[0, 0],
                vx_i[1, 0],
                vx_i[2, 0],
                state.qw,
                state.qx,
                state.qy,
                state.qz,
                state.wx,
                state.wy,
                state.wz,
            ],
            dtype=np.float64,
        )

        # Compute dual quaternion
        self.dual_1 = dualquat_from_pose(
            self.x_0[6],
            self.x_0[7],
            self.x_0[8],
            self.x_0[9],
            self.x_0[0],
            self.x_0[1],
            self.x_0[2],
        )
        # Init linear velocity in the inertial frame and angular velocity in the body frame
        self.angular_linear_1 = np.array(
            [self.x_0[10], self.x_0[11], self.x_0[12], self.x_0[3], self.x_0[4], self.x_0[5]]
        )
        # Init Dual Twist
        self.dual_twist_1 = dual_twist(self.angular_linear_1, self.dual_1)
        self.X[:, 0] = np.array(ca.vertcat(self.dual_1, self.dual_twist_1)).reshape((14,))
        return None

    def send_ref(self):
        self.ref_msg.header.frame_id = "world"
        self.ref_msg.header.stamp = self.get_clock().now().to_msg()

        self.ref_msg.pose.pose.position.x = self.x_ref[0, 0]
        self.ref_msg.pose.pose.position.y = self.x_ref[1, 0]
        self.ref_msg.pose.pose.position.z = self.x_ref[2, 0]

        self.ref_msg.pose.pose.orientation.x = self.x_ref[7, 0]
        self.ref_msg.pose.pose.orientation.y = self.x_ref[8, 0]
        self.ref_msg.pose.pose.orientation.z = self.x_ref[9, 0]
        self.ref_msg.pose.pose.orientation.w = self.x_ref[6, 0]

        # Send Message
        self.publisher_ref_.publish(self.ref_msg)
        return None

    def send_cmd(self, dqd, wd, u):
        control = ControlCommand(
            thrust=float(u[0]),
            torque_x=float(u[1]),
            torque_y=float(u[2]),
            torque_z=float(u[3]),
        )
        self.control_msg = wrench_from_control(control, Wrench)
        self.publisher_control_.publish(self.control_msg)
        return None

    def init_marker(self):
        self.marker_msg.header.frame_id = "world"
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.ns = "trajectory"
        self.marker_msg.id = 0
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        self.marker_msg.pose.orientation.w = 1.0
        self.marker_msg.scale.x = 0.01  # Line width
        self.marker_msg.color.a = 1.0  # Alpha
        self.marker_msg.color.r = 0.0  # Red
        self.marker_msg.color.g = 1.0  # Green
        self.marker_msg.color.b = 0.0  # Blue
        point = Point()
        point.x = self.x_ref[0, 0]
        point.y = self.x_ref[1, 0]
        point.z = self.x_ref[2, 0]
        self.points = [point]
        self.marker_msg.points = self.points
        return None

    def send_marker(self):
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        point = Point()
        point.x = self.x_ref[0, 0]
        point.y = self.x_ref[1, 0]
        point.z = self.x_ref[2, 0]
        self.points.append(point)
        self.marker_msg.points = self.points
        self.publisher_ref_trajectory_.publish(self.marker_msg)
        return None

    def control_nmpc(self):
        # Optimal Control
        self.acados_ocp_solver.set(0, "lbx", self.X[:, 0])
        self.acados_ocp_solver.set(0, "ubx", self.X[:, 0])

        # Desired Trajectory of the system
        for j in range(self.N_prediction):
            yref = self.X_d[:, 0 + j]
            uref = self.u_d[:, 0 + j]
            aux_ref = np.hstack((yref, uref, self.Q, self.Q_e, self.R))
            self.acados_ocp_solver.set(j, "p", aux_ref)

        self.acados_ocp_solver.set(j + 1, "p", aux_ref)
        # Check Solution since there can be possible errors
        self.acados_ocp_solver.solve()
        self.X_control = self.acados_ocp_solver.get(1, "x")
        self.u_control = self.acados_ocp_solver.get(0, "u")
        self.u_aux = np.array(self.u_control)
        self.send_cmd(self.X_control[0:8], self.X_control[8:14], self.u_control)
        # self.get_logger().info(f"Sent control: Thrust={self.u_control[0]:.3f}")
        return None


def main(arg=None):
    rclpy.init(args=arg)
    planning_node = DQnmpcNode()
    try:
        rclpy.spin(planning_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        planning_node.get_logger().info("Simulation stopped manually.")
        planning_node.destroy_node()
        rclpy.shutdown()
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()
    return None


if __name__ == "__main__":
    main()
