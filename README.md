# 2D/3D Spiral Sequences with PyPulseq

[![DOI](https://zenodo.org/badge/723832376.svg)](https://doi.org/10.5281/zenodo.15284022)


## Installation

Create a Python environment with your favorite tool (venv, conda, pyenv). For `venv`:

```python -m venv venv```

Activate your environment. For example, for `venv`:

```source venv/bin/activate```

Install the dependencies:

```pip install -r requirements.txt```

Last command should compile and put the libraries to the necessary places.

*Note:* On some systems (Ubuntu), following package may need to be installed for Qt to work:

`sudo apt install libxcb-cursor0`

### Dependencies
See `requirements.txt`.

`gropt` is optional. If it is not possible to compile it for your system, please use the other rewinder design methods. Note that there are small changes to the Python code of the library as bug fixes.

## Usage

### Configuration

Copy `example_config.toml` and rename it as `config.toml`.

`systems/` contains example scanner specs to copy paste into config.toml [system] part.

Run `write_rtspiral.py` to generate the trajectory and metadata.

### Outputs

When `write_seq = true` in `config.toml`, a `.seq` file will be generated in the `out_seq/` directory, and the corresponding metadata, which will be named as the `sequence_hash.mat` as a `.mat` file will be put into `out_trajectory/` directory.

## References and Acknowledgements

Variable-Density Spiral Design Functions by Brian Hargreaves - http://mrsrl.stanford.edu/~brian/vdspiral/

1. Hoge RD, Kwan RKS, Pike GB. Density compensation functions for spiral MRI. Magnetic Resonance in Medicine. 1997;38(1):117-128. doi:10.1002/mrm.1910380117
2. Loecher M, Middione MJ, Ennis DB. A gradient optimization toolbox for general purpose time-optimal MRI gradient waveform design. Magnetic Resonance in Medicine. 2020;84(6):3234-3245. doi:10.1002/mrm.28384
3. Lee JH, Hargreaves BA, Hu BS, Nishimura DG. Fast 3D imaging using variable-density spiral trajectories with applications to limb perfusion. Magnetic Resonance in Medicine. 2003;50(6):1276-1285. doi:10.1002/mrm.10644
4. Ibrahim ESH, Stuber M, Schär M, Osman NF. Improved myocardial tagging contrast in cine balanced SSFP images. Journal of Magnetic Resonance Imaging. 2006;24(5):1159-1167. doi:10.1002/jmri.20730
5. Chen W, Lee NG, Byrd D, Narayanan S, Nayak KS. Improved real-time tagged MRI using REALTAG. Magnetic Resonance in Medicine. 2020;84(2):838-846. doi:10.1002/mrm.28144
6. Pipe JG, Zwart NR. Spiral trajectory design: A flexible numerical algorithm and base analytical equations. Magnetic Resonance in Medicine. 2014;71(1):278-285. doi:10.1002/mrm.24675

