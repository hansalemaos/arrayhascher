import os
import subprocess
import sys
import numpy as np
import pandas as pd
r = np.frompyfunc(repr, 1, 1)

def _dummyimport():
    import Cython


try:
    from .hasharrycolumn import findduplicates, joinallhashes
except Exception as e:
    cstring = r"""# distutils: language=c++
# distutils: extra_compile_args=/std:c++20 /openmp 
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: sources = xxhash.c
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False
#include xxhash.h

from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython

np.import_array()

cdef extern from "xxhash.h":
    ctypedef unsigned long long XXH64_hash_t
    cdef XXH64_hash_t XXH64(void* input, Py_ssize_t length, XXH64_hash_t seed) nogil

cpdef findduplicates(cython.uchar[:,:]inputarray,Py_ssize_t itemsizex,Py_ssize_t[:,:] outputarray,Py_ssize_t activecolumn,Py_ssize_t seed=1):
    cdef Py_ssize_t inputarrayshape0=inputarray.shape[0]
    cdef Py_ssize_t inputarrayshape1=inputarray.shape[1]
    cdef Py_ssize_t _inputarrayshape0
    for _inputarrayshape0 in prange(inputarrayshape0,nogil=True):
            outputarray[_inputarrayshape0][activecolumn]= XXH64(&inputarray[_inputarrayshape0][0],itemsizex*inputarrayshape1,seed,)

cpdef joinallhashes(cython.uchar[:,:] inputarray, Py_ssize_t[:] outputarray,Py_ssize_t itemsizex,Py_ssize_t seed=1):
    cdef Py_ssize_t inputarrayshape0=inputarray.shape[0]
    cdef Py_ssize_t inputarrayshape1=inputarray.shape[1]
    cdef Py_ssize_t _inputarrayshape0
    cdef Py_ssize_t bytestoread=inputarrayshape1*itemsizex
    for _inputarrayshape0 in prange(inputarrayshape0,nogil=True):
        outputarray[_inputarrayshape0]= XXH64(&inputarray[_inputarrayshape0][0],itemsizex*bytestoread,seed,)
"""
    pyxfile = f"hasharrycolumn.pyx"
    pyxfilesetup = f"hasharrycolumnarraycompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'hasharrycolumn', 'sources': ['hasharrycolumn.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='hasharrycolumn',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .hasharrycolumn import findduplicates, joinallhashes


    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()


def get_hash_column(df,fail_convert_to_string=True,whole_result=False):
    r"""
    Computes a hash value for each column in a DataFrame/NumPy Array/list/tuple.

    Parameters:
    - df (numpy.ndarray, pandas.Series, pandas.DataFrame, list, tuple): 2D (!) Input data to compute hash values for.
    - fail_convert_to_string (bool, optional): If True, tries to convert non-string columns to strings after failed hashing. - The original data won't change!
                                               If False, raises an exception if conversion fails. Default is True.
    - whole_result (bool, optional): If True, returns an array of hash values for each element in the DataFrame/NumPy Array/list/tuple.
                                    If False, returns a condensed array of hash values for each column.
                                    Default is False.

    Returns:
    - numpy.ndarray: If `whole_result` is False, returns a condensed array of hash values for each column.
                     If `whole_result` is True, returns an array of hash values for each element in the DataFrame.

    Example:
        import pandas as pd

        from arrayhascher import get_hash_column

        def test_drop_duplicates(df,hashdata):
            # Example of how to delete duplicates

            return df.assign(__XXXX___DELETE____=hashdata).drop_duplicates(subset='__XXXX___DELETE____').drop(
                columns='__XXXX___DELETE____')

        # With pandas ----------------------------------------------------------------
        df = pd.read_csv(
            "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
        )
        df = pd.concat([df for _ in range(10000)], ignore_index=True)
        df = df.sample(len(df))
        hashdata = get_hash_column(df, fail_convert_to_string=True, whole_result=False)
        # Out[3]:
        # array([-4123592378399267822,   -20629003135630820,  1205215161148196795,
        #        ...,  4571993557129865534, -5454081294880889185,
        #         2672790383060839465], dtype=int64)

        # %timeit test_drop_duplicates(df,hashdata)
        # %timeit df.drop_duplicates()
        # 947 ms ± 18.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # 2.94 s ± 10.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

        # Numpy only  ----------------------------------------------------------------
        hashdata = get_hash_column(df.to_numpy(), fail_convert_to_string=True, whole_result=False)
        print(hashdata)
        # # array([-4123592378399267822,   -20629003135630820,  1205215161148196795,
        # #        ...,  4571993557129865534, -5454081294880889185,
        # #         2672790383060839465], dtype=int64)

        # Works also with lists  ------------------------------------------------------
        get_hash_column(df[:100].to_numpy().tolist(), fail_convert_to_string=True, whole_result=False)
        # array([-5436153420663104440, -1384246600780856199,   177114776690388363,
        #          788413506175135724,  1442743010667139722, -6386366738900951630,
        #        -8610361015858259700,  3995349003546064044,  3627302932646306514,
        #         3448626572271213155, -1555175565302024830,  3265835764424924148, ....
        # And tuples  ----------------------------------------------------------------
        tuple(map(tuple, df[:100].to_numpy().tolist()))

    """
    asuinx = []
    if not isinstance(df,(np.ndarray,pd.Series,pd.DataFrame)):
        try:
            df=np.array(df)
        except Exception:
            df=np.asanyarray(df)
    for i in range(df.shape[1]):
        try:
            colnu = np.ascontiguousarray(df[df.columns[i]].to_numpy())
        except Exception as fe:
            colnu=df[...,i]
        itsi = colnu.itemsize
        try:
            vsw = np.ascontiguousarray(colnu.view(f'V{itsi}'))
        except Exception as fe:
            if not fail_convert_to_string:
                raise fe
            vsw = r(colnu).astype('U')
            vsw = vsw.view(f'V{vsw.itemsize}')
        di = np.ascontiguousarray(vsw.view(np.uint8).reshape((len(colnu), -1)))
        asuinx.append(di)
    outarra = np.zeros(df.shape, dtype=np.int64)
    itemsizex = 1

    for yse in range(len(asuinx)):
        asuin = np.ascontiguousarray(asuinx[yse])
        findduplicates(asuin, itemsizex, outarra, yse)
    if whole_result:
        return outarra
    finaloutput = np.zeros(len(df), dtype=np.int64)
    outarra = np.ascontiguousarray(outarra.view(np.uint8))
    joinallhashes(outarra, finaloutput, outarra.itemsize)
    return finaloutput

