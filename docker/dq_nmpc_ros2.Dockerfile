# dq_nmpc ROS2 adapter — bridges /dev/shm/quadrotor_sim ↔ ROS2 topics
#
# Build:
#   docker build -f docker/dq_nmpc_ros2.Dockerfile -t dq_nmpc_ros2 .
#
# Run (requires sim_core on host):
#   docker run --rm --net=host \
#     -v /dev/shm/quadrotor_sim:/dev/shm/quadrotor_sim:rw \
#     dq_nmpc_ros2

FROM ros:humble-ros-core

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pydantic>=2.0 pyyaml

WORKDIR /ros_ws

COPY docker/ros2_adapter_node.py /ros_ws/adapter_node.py
COPY src/dq_nmpc/schemas/ /ros_ws/dq_nmpc/schemas/
COPY deps/mujoco_quadrotor/python/quadrotor_sim/shm.py /ros_ws/quadrotor_sim/shm.py
COPY deps/mujoco_quadrotor/python/quadrotor_sim/schema.py /ros_ws/quadrotor_sim/schema.py
COPY deps/mujoco_quadrotor/python/quadrotor_sim/__init__.py /ros_ws/quadrotor_sim/__init__.py

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
