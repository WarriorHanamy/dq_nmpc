#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

exec python3 /ros_ws/adapter_node.py "$@"
