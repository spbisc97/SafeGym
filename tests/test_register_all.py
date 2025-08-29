import importlib


def test_register_all_adds_alias_and_is_idempotent():
    import gymnasium as gym
    from safegym import register_all

    # First call should work
    register_all()
    spec = gym.spec("Satellite-SE")
    assert spec is not None

    # Second call should be a no-op (no DuplicateEnv error)
    register_all()
    spec2 = gym.spec("Satellite-SE")
    assert spec2 is not None

