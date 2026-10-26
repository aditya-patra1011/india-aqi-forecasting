import importlib

from src.run_all import STAGES


def test_every_stage_is_an_importable_module_with_main():
    for name, _ in STAGES:
        assert callable(importlib.import_module(f"src.{name}").main), name


def test_stages_run_data_first_and_models_last():
    names = [s for s, _ in STAGES]
    assert names[0] == "make_dataset" and names[-1] == "export_models"
    assert names.index("run_ml") < names.index("run_comparison")      # comparison reads every phase's output
    assert names.index("run_deep") < names.index("run_comparison")
    assert len(set(names)) == len(names)
