# 2D Real-time Spiral Sequences with PyPulseq

## Installation

Create a Python environment with your favorite tool (venv, conda, pyenv). For `venv`:

```python -m venv venv```

Activate your environment. For example, for `venv`:

```source venv/bin/activate```

Install the dependencies:

```pip install -r requirements.txt```

Build `gropt` and `libvds` libraries for your system:

```python setup.py develop```

Last command should compile and put the libraries to the necessary places.

### Dependencies

Refer to `requirements.txt` file.

## Usage

### Configuration

Copy `example_config.toml` and rename it as `config.toml`.

`systems/` contains example scanner specs to copy paste into config.toml [system] part.

Run `write_rtspiral.py` to generate the trajectory and metadata.

### Outputs

When `write_seq = true` in `config.toml`, a `.seq` file will be generated in the `out_seq/` directory, and the corresponding metadata, which will be named as the `sequence_hash.mat` as a `.mat` file will be put into `out_trajectory/` directory.

## References and Acknowledgements

Variable-Density Spiral Design Functions by Brian Hargreaves - http://mrsrl.stanford.edu/~brian/vdspiral/

1. Hoge RD, Kwan RKS, Bruce Pike G. Density compensation functions for spiral MRI. Magnetic Resonance in Medicine. 1997;38(1):117-128. doi:10.1002/mrm.1910380117
2. Loecher M, Middione MJ, Ennis DB. A gradient optimization toolbox for general purpose time-optimal MRI gradient waveform design. Magnetic Resonance in Medicine. 2020;84(6):3234-3245. doi:10.1002/mrm.28384
3. Lee JH, Hargreaves BA, Hu BS, Nishimura DG. Fast 3D imaging using variable-density spiral trajectories with applications to limb perfusion. Magnetic Resonance in Medicine. 2003;50(6):1276-1285. doi:10.1002/mrm.10644

- TODO: Ref to Tagging Papers
