"""SafeGym top-level package.

Provides helper `register_all()` to register all environments with Gymnasium
and add convenient aliases used in examples/README.
"""

from __future__ import annotations

from typing import Any, Dict

import importlib
import warnings


def _maybe_register(id: str, *, entry_point: str, gym_module: Any, **kwargs: Dict[str, Any]) -> None:
    """Register an env if not already present.

    - Avoids duplicate-registration errors when tests or scripts call repeatedly.
    """
    try:
        # If spec exists, do nothing
        gym_module.spec(id)
        return
    except Exception:
        pass

    gym_module.register(id=id, entry_point=entry_point, **kwargs)


def register_all() -> None:
    """Register all SafeGym environments and common aliases.

    This function is safe to call multiple times.
    """
    import gymnasium as gym

    # Import envs subpackage to trigger its registrations (Snake, Satellite-*, etc.).
    try:
        importlib.import_module("safegym.envs")
    except Exception as e:
        warnings.warn(f"safegym.envs import produced warnings/errors: {e}")

    # Add alias used in README/examples: "Satellite-SE" -> Satellite_SE2
    try:
        se2_spec = gym.spec("Satellite-SE2-v0")
        entry = se2_spec.entry_point  # type: ignore[attr-defined]
        # Some Gymnasium versions expose entry as callable or string; normalize to string
        if callable(entry):
            # If entry is a constructor, fall back to known path
            entry_point = "safegym.envs.Satellite_SE2:Satellite_SE2"
        else:
            entry_point = str(entry)
        _maybe_register(
            "Satellite-SE",
            entry_point=entry_point,
            gym_module=gym,
            # Keep it simple; defaults are fine for the alias
        )
    except Exception:
        # If base spec missing, try registering alias directly as a best effort
        _maybe_register(
            "Satellite-SE",
            entry_point="safegym.envs.Satellite_SE2:Satellite_SE2",
            gym_module=gym,
        )


__all__ = ["register_all"]
