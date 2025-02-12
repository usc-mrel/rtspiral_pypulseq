try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import os

def load_params(param_filename:str, param_dir: str = "protocols"):
    fname_ = param_filename + ".toml" if not param_filename.endswith(".toml") else param_filename
    with open(os.path.join(param_dir, fname_), "rb") as f:
        toml_dict = tomllib.load(f)

    return toml_dict