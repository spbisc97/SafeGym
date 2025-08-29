import importlib
import warnings


def test_import_safegym_package():
    mod = importlib.import_module("safegym")
    assert mod is not None


def test_import_envs_module():
    # envs package should import even if optional deps (pygame/mujoco) are missing
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        envs = importlib.import_module("safegym.envs")
    assert envs is not None
    # Required envs should be present
    assert hasattr(envs, "Satellite_SE2")
    assert hasattr(envs, "Satellite_rot")
    assert hasattr(envs, "Satellite_tra")


def test_import_key_env_modules_directly():
    # Direct-import key env modules/classes
    for mod_name, cls_name in [
        ("safegym.envs.Satellite_SE2", "Satellite_SE2"),
        ("safegym.envs.Satellite_rot", "Satellite_rot"),
        ("safegym.envs.Satellite_tra", "Satellite_tra"),
    ]:
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, cls_name)


def test_gym_registration_satellite_se2():
    import gymnasium as gym

    # Import envs to trigger registration side effects
    importlib.import_module("safegym.envs")

    spec = gym.spec("Satellite-SE2-v0")
    assert spec is not None
    assert "Satellite_SE2" in str(spec.entry_point)

