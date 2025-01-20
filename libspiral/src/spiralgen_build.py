from cffi import FFI
import pathlib

src_root = pathlib.Path(__file__).parent.joinpath('ext')
sources = [src_root.joinpath(s).as_posix() for s in ['spiralgen_jgp_12oct.c', 'vds.c']]

ffibuilder = FFI()

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef("""
    void bnispiralgen(float* spparams, int maxarray, float *gxarray, float *gyarray, float *gzarray, 
                  int *spgrad_na, int *spgrad_nb, int *spgrad_nc, int *spgrad_nd);
    void calc_vds(double slewmax, double gradmax, double Tgsample, double Tdsample, int Ninterleaves, double* fov, int numfov, double krmax,
            int ngmax, double** xgrad, double** ygrad, int* numgrad);
""")


# set_source() gives the name of the python extension module to
# produce, and some C source code as a string.  This C code needs
# to make the declarated functions, types and globals available,
# so it is often just the "#include".
ffibuilder.set_source("_spiralgen",
"""
    void bnispiralgen(float* spparams, int maxarray, float *gxarray, float *gyarray, float *gzarray, 
                  int *spgrad_na, int *spgrad_nb, int *spgrad_nc, int *spgrad_nd);
    void calc_vds(double slewmax, double gradmax, double Tgsample, double Tdsample, int Ninterleaves, double* fov, int numfov, double krmax,
		int ngmax, double** xgrad, double** ygrad, int* numgrad);
""",
    sources=sources,   # 
    libraries=['m'])   # library name, for the linker

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

