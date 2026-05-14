"""Domain-specific multi-step pipelines — import lazily to avoid acados dependency."""

__all__ = ["codegen", "generate_trajectory", "nmpc_loop"]


def __getattr__(name):
    if name == "codegen":
        from dq_nmpc.workflows.codegen import codegen as _codegen

        return _codegen
    if name == "generate_trajectory":
        from dq_nmpc.workflows.generate_trajectory import (
            generate_trajectory as _generate_trajectory,
        )

        return _generate_trajectory
    if name == "nmpc_loop":
        from dq_nmpc.workflows.run_nmpc import nmpc_loop as _nmpc_loop

        return _nmpc_loop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
