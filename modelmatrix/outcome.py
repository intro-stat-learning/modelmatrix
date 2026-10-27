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
from .response import Response, Binomial, Multinomial, _values, _output


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


class _Columns(Response):

    # columns of X as they are: the responses of a multi-response model

    def __init__(self, columns):
        self.column_list = list(columns)

    @property
    def columns(self):
        return tuple(self.column_list)

    def _transform(self, X):
        return pd.DataFrame({c: _values(X, c).to_numpy() for c in self.column_list},
                            index=X.index if isinstance(X, pd.DataFrame) else None)

    def __repr__(self):
        return repr(self.column_list)


def _components(response):
    # the responses of a fitted multi-response model, each with its own
    # linear predictor: the columns of a list of columns, the classes of
    # a Multinomial (its levels, whatever its output), else None
    if isinstance(response, _Columns):
        return list(response.column_list)
    if isinstance(response, Multinomial):
        if isinstance(response.classes, list):
            return list(response.classes)
        return list(response.levels_)
    return None


class Outcome(Response):

    """
    A response, with other columns named by their role.

    Parameters
    ----------

    response : Response, column identifier, or list of them
        The response: a :py:class:`Surv`, :py:class:`Binomial` or
        :py:class:`Multinomial`, or a column of `X` as it is, or a list
        of columns, the responses of a multi-response model.

    **columns : column identifier, Feature, or dict of them
        Other columns, each named by its keyword, e.g. `offset='O'` or
        `weight='w'`: a column of `X`, as it is (a categorical, such as
        the strata of a Cox model, keeps its labels), or a feature
        of a single column, e.g. the log of a column, fit as a term of a
        :py:class:`ModelSpec`.

        For a multi-response model, a dict gives one column for each
        response, e.g. an offset of each linear predictor: its keys are
        the response's columns, or the classes of a
        :py:class:`Multinomial` (whatever its output), and
        `offset={'y1': 'O1', 'y2': 'O2'}` gives the columns
        `offset[y1]` and `offset[y2]`, in the order of the responses.
        A single column, `offset='O'`, is the same for every response.

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
            values = value.values() if isinstance(value, dict) else [value]
            for v in values:
                if isinstance(v, Feature):
                    used = _term_columns([v])
                    if used is None:
                        raise ValueError(f'the column {name!r} of an Outcome cannot use a wildcard')
                    cols.extend(sorted(used, key=str))
                else:
                    cols.append(v)
        return tuple(dict.fromkeys(cols))

    def _fit(self, X):
        self.response_ = deepcopy(_as_response(self.response))
        self.response_._fit(X)
        self.response_.columns_ = self.response_.columns
        # the other columns, by their names: a dict gives one for each
        # response, in their order
        self.entries_ = {}
        for name, value in self.extra.items():
            if isinstance(value, dict):
                components = _components(self.response_)
                if components is None:
                    raise ValueError(f'{name!r} is a dict, one column for each response, '
                                     f'but {self.response!r} is not a multi-response')
                missing = [c for c in components if c not in value]
                unknown = [k for k in value if k not in components]
                if missing or unknown:
                    raise ValueError(f'the keys of {name!r} should be the responses {components}: '
                                     f'missing {missing}, unknown {unknown}')
                for c in components:
                    self.entries_[f'{name}[{c}]'] = value[c]
            else:
                self.entries_[name] = value
        self.specs_ = {}
        for name, value in self.entries_.items():
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
        for name, value in self.entries_.items():
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
        def show(v):
            if isinstance(v, dict):
                return '{' + ', '.join(f'{k!r}: {show(u)}' for k, u in v.items()) + '}'
            return v.name if isinstance(v, Feature) else repr(v)
        args = [repr(self.response)] + [f'{k}={show(v)}' for k, v in self.extra.items()]
        return f'Outcome({", ".join(args)})'


def _as_response(response):
    if isinstance(response, Response):
        return response
    if isinstance(response, list):
        return _Columns(response)
    return _Column(response)
