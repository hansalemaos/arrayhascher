# Fast hash in 2D Arrays (Numpy/Pandas/lists/tuples)

## pip install arrayhascher

### Tested against Windows / Python 3.11 / Anaconda


## Cython (and a C/C++ compiler) must be installed



```python
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
```