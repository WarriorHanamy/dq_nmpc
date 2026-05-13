"""ROS2 adapter: bridges /dev/shm/quadrotor_sim to ROS2 topics.

Publishes /odom (nav_msgs/Odometry) from SHM state.
Subscribes to /cmd (geometry_msgs/Wrench) and writes to SHM control.
"""

import rclpy
from geometry_msgs.msg import Wrench
from nav_msgs.msg import Odometry
from rclpy.node import Node

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from quadrotor_sim.shm import (
    SHM_CTRL_FILE,
    SHM_STATE_FILE,
    QuadrotorControlC,
    QuadrotorStateC,
    ShmReader,
    ShmWriter,
)

# re-expose schemas at the expected import path
import dq_nmpc.schemas.state  # noqa: F401


class DqNmpcRosAdapter(Node):
    def __init__(self, namespace="quadrotor"):
        super().__init__("dq_nmpc_ros_adapter", namespace=namespace)

        self.declare_parameter("rate_odom", 100.0)
        self.declare_parameter("rate_cmd", 100.0)

        self._state_reader = ShmReader(SHM_STATE_FILE, QuadrotorStateC, 192)
        self._ctrl_writer = ShmWriter(SHM_CTRL_FILE, QuadrotorControlC, 64)
        self._state_buf = QuadrotorStateC()

        self._state_reader.attach()
        try:
            self._ctrl_writer.open()
        except FileNotFoundError:
            self._ctrl_writer.create()

        self._odom_pub = self.create_publisher(Odometry, "odom", 10)
        self._cmd_sub = self.create_subscription(Wrench, "cmd", self._on_cmd, 10)

        odom_rate = self.get_parameter("rate_odom").value
        self._odom_timer = self.create_timer(1.0 / odom_rate, self._publish_odom)

        self.get_logger().info("ROS2 adapter started (odom=%d Hz)", odom_rate)

    def _publish_odom(self):
        if not self._state_reader.read(self._state_buf):
            return
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.child_frame_id = "base_link"
        p = self._state_buf.position
        o = self._state_buf.orientation
        lv = self._state_buf.linear_velocity
        av = self._state_buf.angular_velocity
        msg.pose.pose.position.x = p[0]
        msg.pose.pose.position.y = p[1]
        msg.pose.pose.position.z = p[2]
        msg.pose.pose.orientation.w = o[0]
        msg.pose.pose.orientation.x = o[1]
        msg.pose.pose.orientation.y = o[2]
        msg.pose.pose.orientation.z = o[3]
        msg.twist.twist.linear.x = lv[0]
        msg.twist.twist.linear.y = lv[1]
        msg.twist.twist.linear.z = lv[2]
        msg.twist.twist.angular.x = av[0]
        msg.twist.twist.angular.y = av[1]
        msg.twist.twist.angular.z = av[2]
        self._odom_pub.publish(msg)

    def _on_cmd(self, msg: Wrench):
        self._ctrl_writer.write_control(
            thrust=msg.force.z,
            torque_x=msg.torque.x,
            torque_y=msg.torque.y,
            torque_z=msg.torque.z,
        )

    def destroy_node(self):
        self._state_reader.detach()
        self._ctrl_writer.detach()
        super().destroy_node()


def main():
    rclpy.init()
    ns = os.environ.get("ROS_NAMESPACE", "quadrotor")
    node = DqNmpcRosAdapter(namespace=ns)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
