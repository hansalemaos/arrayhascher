# Parallel argsort with Cython - 2.5x faster than NumPy!

## pip install cythonparallelargsort 

### Tested against Windows / Python 3.11 / Anaconda

## Cython (and a C/C++ compiler) must be installed

A parallel argsort function for efficiently sorting numpy arrays in parallel.
It utilizes Cython to generate optimized C++ code, taking advantage of OpenMP for parallelism.

```python
from cythonparallelargsort import parallel_argsort
import pandas as pd
import numpy as np
df = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
)
df=pd.concat([df for _ in range(1000)],ignore_index=True)
df=df.sample(len(df))
indicopy=df.index.to_numpy().copy()
p1=parallel_argsort(indicopy,'parallel_buffered',)
p2=parallel_argsort(indicopy,'parallel',)
p3=parallel_argsort(indicopy,'sort')
p4=np.argsort(indicopy)

# p1
# Out[3]: array([125054,  85353, 788878, ...,  46414, 789033, 786844], dtype=int64)
# p2
# Out[4]: array([125054,  85353, 788878, ...,  46414, 789033, 786844], dtype=int64)
# p3
# Out[5]: array([125054,  85353, 788878, ...,  46414, 789033, 786844], dtype=int64)
# p4
# Out[6]: array([125054,  85353, 788878, ...,  46414, 789033, 786844], dtype=int64)


# df.index.shape
# Out[8]: (89100,)
# %timeit p1=parallel_argsort(indicopy,'parallel_buffered',)
# %timeit p2=parallel_argsort(indicopy,'parallel',)
# %timeit p3=parallel_argsort(indicopy,'sort')
# %timeit p4=np.argsort(indicopy)
# 3.18 ms ± 622 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 2.62 ms ± 38.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 6.38 ms ± 54.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 5.13 ms ± 99.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# df.index.shape
# (891000,)
# %timeit p1=parallel_argsort(indicopy,'parallel_buffered',)
# %timeit p2=parallel_argsort(indicopy,'parallel',)
# %timeit p3=parallel_argsort(indicopy,'sort')
# %timeit p4=np.argsort(indicopy)
# 28.4 ms ± 2.42 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 29 ms ± 1.04 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 90.9 ms ± 1.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 73.3 ms ± 910 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# df.index.shape
# Out[4]: (8910000,)
# %timeit p1=parallel_argsort(indicopy,'parallel_buffered',)
# %timeit p2=parallel_argsort(indicopy,'parallel',)
# %timeit p3=parallel_argsort(indicopy,'sort')
# %timeit p4=np.argsort(indicopy)
# 586 ms ± 24.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 604 ms ± 18.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 1.45 s ± 20.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 1.34 s ± 13.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# df.index.shape
# Out[4]: (89100000,)
# %timeit p1=parallel_argsort(indicopy,'parallel_buffered',)
# %timeit p2=parallel_argsort(indicopy,'parallel',)
# %timeit p3=parallel_argsort(indicopy,'sort')
# %timeit p4=np.argsort(indicopy)
# 10.1 s ± 97.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 10.5 s ± 45.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 20.9 s ± 82.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 24.5 s ± 70.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```