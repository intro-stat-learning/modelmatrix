"""
Outcome
=======

A response with other columns a model takes with it, such as an offset,
weights or the strata of a Cox model, each named by its role:
`Outcome(Surv('time', 'status'), offset='O', weight='w')` builds a
dataframe with columns `stop`, `status`, `offset` and `weight`, so that
an estimator can find each column by its name.
"""

from copy import deepcopy

import numpy as np, pandas as pd

from ._native import _from_native
from .model_spec import ModelSpec, Feature, _term_columns
from .response import Response, Binomial, _values, _output


class _Column(Response):

    # a column of X as it is

    def __init__(self, column):
        self.column = column

    @property
    def columns(self):
        return (self.column,)

    def _transform(self, X):
        return _values(X, self.column).copy()

    def __repr__(self):
        return repr(self.column)


class Outcome(Response):

    """
    A response, with other columns named by their role.

    Parameters
    ----------

    response : Response or column identifier
        The response: a :py:class:`Surv`, :py:class:`Binomial` or
        :py:class:`Multinomial`, or a column of `X` as it is.

    **columns : column identifier or Feature
        Other columns, each named by its keyword, e.g. `offset='O'` or
        `weight='w'`: a column of `X`, as it is (a categorical, such as
        the strata of a Cox model, keeps its labels), or a feature
        of a single column, e.g. the log of a column, fit as a term of a
        :py:class:`ModelSpec`.

    Notes
    -----

    The columns are those of the response, then the others in the order
    of their keywords. With a :py:class:`Binomial` of
    `output='proportion'`, its number of trials is the column `weight`
    (times the weights given as `weight=`, if any), as R's prior weights
    of a proportion.

    `wildcard(exclude=outcome)` leaves out every column the outcome uses,
    so that none of them is a term of its own design. For an offset of
    new data, without its response, use `transform(X, response=False)`.
    """

    def __init__(self, response, **columns):
        self.response = response
        self.extra = columns

    @property
    def columns(self):
        cols = list(_as_response(self.response).columns)
        for name, value in self.extra.items():
            if isinstance(value, Feature):
                used = _term_columns([value])
                if used is None:
                    raise ValueError(f'the column {name!r} of an Outcome cannot use a wildcard')
                cols.extend(sorted(used, key=str))
            else:
                cols.append(value)
        return tuple(dict.fromkeys(cols))

    def _fit(self, X):
        self.response_ = deepcopy(_as_response(self.response))
        self.response_._fit(X)
        self.response_.columns_ = self.response_.columns
        self.specs_ = {}
        for name, value in self.extra.items():
            if isinstance(value, Feature):
                spec = ModelSpec([value]).fit(X)
                if len(spec.get_feature_names_out()) != 1:
                    raise ValueError(f'the column {name!r} of an Outcome should be a single '
                                     f'column: {value.name!r} has '
                                     f'{len(spec.get_feature_names_out())}')
                self.specs_[name] = spec
            else:
                _values(X, value)   # raises if the column is not in X
        self._build(X, response=True)   # checks the names

    def transform(self, X, response=True):
        """
        Build the outcome from `X`, after fitting.

        Parameters
        ----------
        X : dataframe or np.ndarray
            Data holding the columns of the outcome.

        response : bool, default=True
            Build the response? If False, only the other columns, e.g.
            the offset of new data, whose response is not known.

        Returns
        -------
        y : dataframe or np.ndarray
            Of the backend of *X*, an :py:class:`np.ndarray` for an array.
        """
        self._check_fitted()
        X, to_native = _from_native(X, set(self.columns))
        y = self._build(X, response)
        return y if to_native is None else to_native(y)

    def _transform(self, X):
        return self._build(X, response=True)

    def _build(self, X, response):
        X_orig = X
        if not isinstance(X, pd.DataFrame):   # an array: its columns are integers
            X = pd.DataFrame(np.asarray(X))
        parts = []
        weight = None
        if response:
            y = self.response_._transform(X)
            y = y.to_frame() if isinstance(y, pd.Series) else y
            if (isinstance(self.response_, Binomial) and self.response_.trials is not None
                and self.response_.output == 'proportion'):
                # the trials are the weights of a proportion
                weight = y.pop(self.response_.trials)
            parts.append(y)
        extra = {}
        for name, value in self.extra.items():
            if name in self.specs_:
                D = self.specs_[name].transform(X)
                extra[name] = np.asarray(D, dtype=float)[:, 0]
            else:
                extra[name] = _values(X, value).copy()   # keeps a categorical
        if weight is not None:
            w = weight.to_numpy()
            extra['weight'] = w if 'weight' not in extra else w * np.asarray(extra['weight'], float)
        parts.append(pd.DataFrame(extra, index=X.index))
        D = pd.concat(parts, axis=1)
        duplicated = list(D.columns[D.columns.duplicated()])
        if duplicated:
            raise ValueError(f'the columns of the Outcome have the same names {duplicated}: '
                             'a column of the response has the name of another')
        return _output(D, X_orig)

    def __repr__(self):
        args = [repr(self.response)] + [f'{k}={v.name if isinstance(v, Feature) else repr(v)}'
                                        for k, v in self.extra.items()]
        return f'Outcome({", ".join(args)})'


def _as_response(response):
    if isinstance(response, Response):
        return response
    return _Column(response)
