from cffi import FFI

ffibuilder = FFI()

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef("""
   void calc_vds(double slewmax, double gradmax, double Tgsample, double Tdsample, int Ninterleaves, double* fov, int numfov, double krmax,
		int ngmax, double** xgrad, double** ygrad, int* numgrad);
""")


# set_source() gives the name of the python extension module to
# produce, and some C source code as a string.  This C code needs
# to make the declarated functions, types and globals available,
# so it is often just the "#include".
ffibuilder.set_source("libvds._vds",
"""
   void calc_vds(double slewmax, double gradmax, double Tgsample, double Tdsample, int Ninterleaves, double* fov, int numfov, double krmax,
		int ngmax, double** xgrad, double** ygrad, int* numgrad);
""",
    sources=['libvds/vds.c'],   # 
    libraries=['m'])   # library name, for the linker

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

