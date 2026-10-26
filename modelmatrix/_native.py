"""
Dataframes of other backends
============================

A dataframe of any eager backend supported by narwhals, e.g. polars or
pyarrow, is converted to pandas, and the results are converted back.
"""

import numpy as np, pandas as pd
import narwhals as nw
from scipy import sparse as sp_sparse


def _series_to_native(S, backend):
    # a categorical is an Enum, if the backend has one, of its
    # categories as strings
    values, dtype = S.to_numpy(), None
    if isinstance(S.dtype, pd.CategoricalDtype):
        values = np.asarray([None if pd.isna(v) else str(v) for v in S], dtype=object)
        dtype = nw.Enum([str(c) for c in S.cat.categories])
    try:
        return nw.new_series(str(S.name), values, dtype=dtype, backend=backend).to_native()
    except NotImplementedError:   # e.g. no Enum in pyarrow
        return nw.new_series(str(S.name), values, backend=backend).to_native()

def _from_native(X, columns=None):
    """
    Convert a dataframe (or series) of any eager backend supported by
    narwhals, e.g. polars or pyarrow, to pandas. Returns the converted
    `X` and a function converting a `pd.DataFrame` model matrix back to
    that backend, or None if `X` was not converted.

    If `columns` is given, only those columns of a dataframe are
    converted (the copy is of the columns a model matrix uses).

    Conversion through pandas keeps categoricals, including the order
    of a polars Enum or an ordered pyarrow dictionary (as an ordered
    `pd.Categorical`). The levels of an unordered categorical, e.g. a
    polars Categorical, are stored in order of appearance; they are
    sorted, as pandas does when it infers categories, so that the
    model matrix does not depend on the order of the rows.
    """
    if (isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)) or
        sp_sparse.issparse(X)):
        return X, None
    try:
        X_nw = nw.from_native(X, allow_series=True)
    except TypeError:   # not a dataframe, e.g. a list: left to _check_X
        return X, None
    if isinstance(X_nw, nw.LazyFrame):
        raise TypeError('Lazy frames are not supported by ModelSpec; '
                        'collect them first, e.g. X.collect()')
    backend = nw.get_native_namespace(X_nw)
    if columns is not None and isinstance(X_nw, nw.DataFrame):
        keep = [c for c in X_nw.columns if c in columns]
        X_nw = X_nw.select(keep or X_nw.columns[:1])

    def to_native(D):
        if isinstance(D, pd.Series):   # e.g. a response
            return _series_to_native(D, backend)
        duplicated = list(D.columns[D.columns.duplicated()])
        if duplicated:
            raise ValueError(f'the model matrix has duplicated column names {duplicated}, '
                             'which this dataframe backend cannot hold')
        return nw.from_dict({str(c): D[c].to_numpy() for c in D.columns},
                            backend=backend).to_native()

    X = X_nw.to_pandas()
    if isinstance(X_nw, nw.Series):
        dtypes = {X.name: X_nw.dtype}
    else:
        dtypes = dict(X_nw.schema)
    for name, dtype in dtypes.items():
        col = X if isinstance(X, pd.Series) else X[name]
        if not isinstance(col.dtype, pd.CategoricalDtype):
            continue
        # some versions of polars convert an Enum to an unordered
        # pd.Categorical: its categories are in the declared order
        if dtype == nw.Enum:
            col = col.cat.as_ordered()
        elif not col.dtype.ordered:
            col = col.cat.reorder_categories(sorted(col.dtype.categories))
        else:
            continue
        if isinstance(X, pd.Series):
            X = col
        else:
            X[name] = col
    return X, to_native
