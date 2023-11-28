try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import os

def load_params(param_filename:str, param_dir: str = "protocols"):
    with open(os.path.join(param_dir, param_filename+".toml"), "rb") as f:
        toml_dict = tomllib.load(f)

    return toml_dict