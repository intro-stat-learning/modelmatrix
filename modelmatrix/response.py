"""
Responses
=========

The response of a model, built from the columns of `X` as a
`ModelSpec` builds its design: `ModelSpec(terms, response=...)`,
then `spec.build_response(X)`. A response is a column (or a list of
columns) of `X`, or one of

- `Surv`: a survival response, as R's `Surv(time, status)` or
  `Surv(start, stop, status)`;
- `Binomial`: a binary response, or successes out of a number of
  trials;
- `Multinomial`: the class of each observation, or counts of
  each class.

A response is fit on the `X` the `ModelSpec` is fit on, e.g. to
learn the levels of a categorical, so that it is coded the same way
on new data.
"""

import numpy as np, pandas as pd


class Response(object):

    """
    Base class of the responses: the columns of `X` used, `fit`
    and `transform`.
    """

    columns = ()

    def fit(self, X):
        for col in self.columns:
            _values(X, col)   # raises if the column is not in X
        return self

    def transform(self, X):
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(map(repr, self.columns))})'


def _values(X, col):
    # a column of X, as a pd.Series
    if isinstance(X, pd.DataFrame):
        if col not in X.columns:
            raise ValueError(f'the response column {col!r} is not in X')
        return X[col]
    X = np.asarray(X)
    if not isinstance(col, (int, np.integer)) or not 0 <= col < X.shape[1]:
        raise ValueError(f'the response column {col!r} is not a column of X, '
                         f'an array with {X.shape[1]} columns')
    return pd.Series(X[:, col], name=col)


def _numeric(X, col):
    values = _values(X, col)
    try:
        return values.astype(float)
    except (TypeError, ValueError):
        raise ValueError(f'the response column {col!r} should be numeric, '
                         f'got dtype {values.dtype}')


def _output(D, X):
    # pandas for a pd.DataFrame X, else np.ndarray
    if isinstance(X, pd.DataFrame):
        return D
    return D.to_numpy()


class _Columns(Response):

    # a column, or a list of columns, of X as they are

    def __init__(self, columns, single):
        self.columns = tuple(columns)
        self.single = single

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if self.single:
                return X[self.columns[0]].copy()
            return X[list(self.columns)].copy()
        X = np.asarray(X)
        if self.single:
            return X[:, self.columns[0]]
        return X[:, list(self.columns)]

    def __repr__(self):
        if self.single:
            return repr(self.columns[0])
        return repr(list(self.columns))


def _as_response(response):
    # a Response for the `response` parameter of ModelSpec
    if response is None or isinstance(response, Response):
        return response
    if isinstance(response, list):
        return _Columns(response, single=False)
    return _Columns((response,), single=True)


class Surv(Response):

    """
    A survival response, as R's `Surv`: `Surv(time, status)` for
    right-censored data, `Surv(start, stop, status)` for (start, stop]
    intervals (counting process data), or with keywords
    `Surv(stop=..., status=..., start=...)`.

    Parameters
    ----------

    stop : column identifier
        Times of the events, or of censoring.

    status : column identifier
        1 (or True) for an event, 0 (or False) for censoring.

    start : column identifier (optional)
        Start of the interval at risk.

    Notes
    -----

    The response is a :py:class:`pd.DataFrame` with columns `start`
    (if given), `stop` and `status`, named as in `X`, e.g. for glmstar's
    `CoxFamily(event_id=stop, status_id=status, start_id=start)` or for
    lifelines' `CoxPHFitter`.
    """

    def __init__(self, *args, start=None, stop=None, status=None):
        if args:
            if start is not None or stop is not None or status is not None:
                raise TypeError('give the columns of Surv by position or by keyword, not both')
            if len(args) == 2:
                stop, status = args
            elif len(args) == 3:
                start, stop, status = args
            else:
                raise TypeError('Surv takes (time, status) or (start, stop, status), '
                                f'got {len(args)} columns')
        if stop is None or status is None:
            raise TypeError('Surv needs the columns of the times (stop) and of the status')
        self.start, self.stop, self.status = start, stop, status

    @property
    def columns(self):
        if self.start is None:
            return (self.stop, self.status)
        return (self.start, self.stop, self.status)

    def fit(self, X):
        self.transform(X)   # checks the columns and their values
        return self

    def transform(self, X):
        status = _numeric(X, self.status)   # booleans as 0 and 1
        observed = status[status.notna()]
        if not observed.isin([0, 1]).all():
            raise ValueError(f'the status {self.status!r} of Surv should be 0 or 1 (or boolean), '
                             f'got the values {sorted(set(observed) - {0, 1})[:5]}')
        stop = _numeric(X, self.stop)
        D = {}
        if self.start is not None:
            start = _numeric(X, self.start)
            if (start >= stop).any():
                raise ValueError(f'the start {self.start!r} of Surv should be less than '
                                 f'its stop {self.stop!r}')
            D[self.start] = start.to_numpy()
        D[self.stop] = stop.to_numpy()
        D[self.status] = status.to_numpy()
        index = X.index if isinstance(X, pd.DataFrame) else None
        return _output(pd.DataFrame(D, index=index), X)

    def __repr__(self):
        if self.start is None:
            return f'Surv({self.stop!r}, {self.status!r})'
        return f'Surv({self.start!r}, {self.stop!r}, {self.status!r})'


class Binomial(Response):

    """
    A binomial response: a binary column, or the number of successes
    out of a number of trials.

    Parameters
    ----------

    successes : column identifier
        A binary column (0/1, boolean, or with two levels), or with
        *trials*, the number of successes.

    trials : column identifier (optional)
        The number of trials.

    level : level identifier (optional)
        For a binary column: the level counted as a success, any other
        level being a failure. Defaults to the second level, as in R: of
        the categories of a categorical, else of the sorted values.

    output : {'counts', 'proportion'}
        With *trials*: `'counts'` gives the successes and the failures,
        two columns as R's `cbind(successes, failures)` (e.g. for
        statsmodels' `GLM` with a `Binomial` family); `'proportion'` the
        proportion of successes and the number of trials, as a response
        and its weights (e.g. for glmstar's `response_id` and `weight_id`).

    Notes
    -----

    A binary column gives a :py:class:`pd.Series` of 0.0 and 1.0
    (missing values are `NaN`); with *trials*, a :py:class:`pd.DataFrame`.
    """

    def __init__(self, successes, trials=None, level=None, output='counts'):
        self.successes = successes
        self.trials = trials
        self.level = level
        self.output = output

    @property
    def columns(self):
        if self.trials is None:
            return (self.successes,)
        return (self.successes, self.trials)

    def fit(self, X):
        if self.trials is not None:
            if self.output not in ('counts', 'proportion'):
                raise ValueError(f"output should be 'counts' or 'proportion', got {self.output!r}")
            self.transform(X)   # checks the values
            return self
        values = _values(X, self.successes)
        observed = values[values.notna()]
        if isinstance(values.dtype, pd.CategoricalDtype):
            levels = list(values.cat.categories)
        elif values.dtype == bool:
            levels = [False, True]
        elif pd.api.types.is_numeric_dtype(values.dtype):
            levels = [0, 1]
            if self.level is None and not observed.isin([0, 1]).all():
                raise ValueError(f'the binary response {self.successes!r} should be 0 or 1, '
                                 f'got the values {sorted(set(observed) - {0, 1})[:5]}')
        else:
            levels = sorted(observed.unique())
        if self.level is not None:
            if self.level not in levels and self.level not in set(observed):
                raise ValueError(f'the level {self.level!r} is not a level of {self.successes!r}')
            self.level_ = self.level
        elif len(levels) != 2:
            raise ValueError(f'the binary response {self.successes!r} should have two levels, '
                             f'got {len(levels)}: {levels[:5]}; give the level '
                             'of a success with Binomial(..., level=...)')
        else:
            self.level_ = levels[1]
        self.levels_ = levels
        return self

    def transform(self, X):
        index = X.index if isinstance(X, pd.DataFrame) else None
        if self.trials is None:
            values = _values(X, self.successes)
            missing = values.isna().to_numpy()
            if self.level is None:
                unseen = set(values[~missing]) - set(self.levels_)
                if unseen:
                    raise ValueError(f'the binary response {self.successes!r} has levels '
                                     f'not seen at fit: {sorted(unseen, key=str)[:5]}')
            y = (values == self.level_).to_numpy().astype(float)
            y[missing] = np.nan
            return _output(pd.Series(y, index=index, name=self.successes), X)

        s = _numeric(X, self.successes).to_numpy()
        n = _numeric(X, self.trials).to_numpy()
        with np.errstate(invalid='ignore'):
            if ((s < 0) | (s > n)).any():
                raise ValueError(f'the successes {self.successes!r} should be between 0 '
                                 f'and the trials {self.trials!r}')
        if self.output == 'counts':
            D = pd.DataFrame({self.successes: s,
                              f'{self.trials}-{self.successes}': n - s}, index=index)
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                D = pd.DataFrame({f'{self.successes}/{self.trials}': s / n,
                                  self.trials: n}, index=index)
        return _output(D, X)

    def __repr__(self):
        args = [repr(self.successes)]
        if self.trials is not None:
            args.append(f'trials={self.trials!r}')
        if self.level is not None:
            args.append(f'level={self.level!r}')
        if self.output != 'counts':
            args.append(f'output={self.output!r}')
        return f'Binomial({", ".join(args)})'


class Multinomial(Response):

    """
    A multinomial response: the class of each observation, or the
    counts of each class.

    Parameters
    ----------

    classes : column identifier, or list of column identifiers
        A column holding the class of each observation, or a column
        of counts for each class.

    output : {'labels', 'onehot'}
        For a column of classes: `'labels'` gives the classes as a
        categorical, its categories the levels seen at fit (e.g. for a
        scikit-learn classifier); `'onehot'` a column of indicators for
        each level.

    Notes
    -----

    The levels are the categories of a categorical, else the sorted
    values. A level not seen at fit raises an error.
    """

    def __init__(self, classes, output='labels'):
        self.classes = classes
        self.output = output

    @property
    def columns(self):
        if isinstance(self.classes, list):
            return tuple(self.classes)
        return (self.classes,)

    def fit(self, X):
        if isinstance(self.classes, list):
            self.transform(X)   # checks the counts
            return self
        if self.output not in ('labels', 'onehot'):
            raise ValueError(f"output should be 'labels' or 'onehot', got {self.output!r}")
        values = _values(X, self.classes)
        if isinstance(values.dtype, pd.CategoricalDtype):
            self.levels_ = list(values.cat.categories)
        else:
            self.levels_ = sorted(values[values.notna()].unique())
        return self

    def transform(self, X):
        index = X.index if isinstance(X, pd.DataFrame) else None
        if isinstance(self.classes, list):
            D = pd.DataFrame({c: _numeric(X, c).to_numpy() for c in self.classes},
                             index=index)
            if (D < 0).any().any():
                raise ValueError(f'the counts {self.classes} of Multinomial should be nonnegative')
            return _output(D, X)
        values = _values(X, self.classes)
        missing = values.isna().to_numpy()
        codes = pd.Index(self.levels_).get_indexer(np.asarray(values, dtype=object))
        unseen = (codes < 0) & ~missing
        if unseen.any():
            raise ValueError(f'the classes {self.classes!r} have levels not seen at fit: '
                             f'{sorted(set(np.asarray(values)[unseen]), key=str)[:5]}')
        if self.output == 'labels':
            labels = pd.Categorical.from_codes(codes, categories=self.levels_)
            return _output(pd.Series(labels, index=index, name=self.classes), X)
        D = np.equal.outer(codes, np.arange(len(self.levels_))).astype(float)
        D[missing] = np.nan
        return _output(pd.DataFrame(D, index=index,
                                    columns=[f'{self.classes}[{l}]' for l in self.levels_]), X)

    def __repr__(self):
        if self.output != 'labels':
            return f'Multinomial({self.classes!r}, output={self.output!r})'
        return f'Multinomial({self.classes!r})'
