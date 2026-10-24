"""Import a script from scripts/ by name (scripts/ is not a package)."""
import importlib.util
import pathlib


def load_script(name):
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
