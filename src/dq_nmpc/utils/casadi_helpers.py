"""CasADi quaternion utility functions."""

from __future__ import annotations

import warnings
from pathlib import Path

import casadi as ca
import yaml

from dq_nmpc.type import Quaternion, Vec3, VecN


def calc_quat_cost(
    q_des: Quaternion,
    q_cur: Quaternion,
    weight: VecN,
) -> ca.MX | ca.SX:
    diff = multiply_quaternions(q_des, conjugate_quaternion(q_cur))
    cost = ca.transpose(diff) @ ca.diag(weight) @ diff
    return cost


def calc_vec_cost(
    des: VecN,
    curr: VecN,
    weight: VecN,
) -> ca.MX | ca.SX:
    diff = curr - des
    cost = ca.transpose(diff) @ ca.diag(weight) @ diff
    return cost


def normalize_quaternion(q: Quaternion) -> ca.MX | ca.SX:
    magnitude = ca.norm_2(q)
    return q / magnitude


def conjugate_quaternion(q: Quaternion) -> ca.MX | ca.SX:
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])


def multiply_quaternions(q1: Quaternion, q2: Quaternion) -> ca.MX | ca.SX:
    qw = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    qx = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    qy = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    qz = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return ca.vertcat(qw, qx, qy, qz)


def rotate_vector_by_quaternion(vec: Vec3, q: Quaternion) -> ca.MX | ca.SX:
    p = ca.vertcat(0, vec)
    q_conjugate = conjugate_quaternion(q)
    rotated_vec = multiply_quaternions(multiply_quaternions(q, p), q_conjugate)
    return rotated_vec[1:]


def yaml_to_dict(path_to_yaml: str | Path) -> dict:
    warnings.warn(
        "yaml_to_dict is deprecated; use NMPCConfig.from_yaml() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    with open(path_to_yaml, "r") as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    if "/**" in parsed_yaml:
        parsed_yaml = parsed_yaml["/**"]["ros__parameters"]
    return parsed_yaml
