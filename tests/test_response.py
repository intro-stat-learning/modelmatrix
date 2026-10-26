# Responses: Surv, Binomial and Multinomial, fit and built from a dataframe

import numpy as np, pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from modelmatrix import ModelSpec as MS, wildcard, Surv, Binomial, Multinomial

from _datasets import load_data


@pytest.fixture(scope='module')
def Carseats():
    return load_data('Carseats')


@pytest.fixture
def S():
    rng = np.random.default_rng(1)
    n = 40
    start = rng.uniform(size=n)
    return pd.DataFrame({'start': start,
                         'time': start + rng.exponential(size=n),
                         'status': rng.binomial(1, 0.7, n),
                         'x': rng.standard_normal(n),
                         'cases': rng.binomial(10, 0.3, n),
                         'size': np.full(n, 10),
                         'flag': rng.binomial(1, 0.5, n).astype(bool),
                         'kind': pd.Categorical(rng.choice(['lo', 'mid', 'hi'], n),
                                                categories=['lo', 'mid', 'hi'])})


def test_not_fit_or_missing_column(S):
    with pytest.raises(NotFittedError):
        Surv('time', 'status').transform(S)
    with pytest.raises(ValueError, match="'nope' is not in X"):
        Binomial('nope').fit(S)


def test_with_a_design(Carseats):
    # R's glm(US ~ ., family=binomial, data=Carseats)
    response = Binomial('US')
    y = response.fit_transform(Carseats)
    spec = MS([wildcard(exclude=response)], intercept=True)
    X = spec.fit_transform(Carseats)
    assert 'US[Yes]' not in X.columns
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    assert np.all(np.isfinite(fit.params))
    scores = cross_val_score(make_pipeline(MS([wildcard(exclude='Sales')]), LinearRegression()),
                             Carseats, Carseats['Sales'], cv=3, error_score='raise')
    assert np.all(np.isfinite(scores))


def test_index_of_y(Carseats):
    spec = MS(['Price'])
    spec.fit(Carseats, Carseats['Sales'])
    with pytest.raises(ValueError, match='indices of X and y do not agree'):
        spec.fit(Carseats, Carseats['Sales'].reset_index(drop=True).iloc[::-1])
    # arrays are not checked
    spec.fit(Carseats, np.asarray(Carseats['Sales']))


# Surv

def test_surv(S):
    y = Surv('time', 'status').fit_transform(S)
    pd.testing.assert_frame_equal(y, S[['time', 'status']].astype(float))
    y3 = Surv('start', 'time', 'status').fit_transform(S)
    assert list(y3.columns) == ['start', 'time', 'status']
    y_kw = Surv(stop='time', status='status', start='start').fit_transform(S)
    pd.testing.assert_frame_equal(y3, y_kw)
    assert repr(Surv('start', 'time', 'status')) == "Surv('start', 'time', 'status')"
    # a boolean status
    yb = Surv('time', 'flag').fit_transform(S)
    np.testing.assert_array_equal(yb['flag'], S['flag'].astype(float))


def test_surv_matches_statsmodels_PHReg(S):
    response = Surv('time', 'status')
    y = response.fit_transform(S)
    X = MS([wildcard(exclude=[response, 'start', 'cases', 'size', 'flag', 'kind'])]).fit_transform(S)
    assert list(X.columns) == ['x']
    fit = sm.PHReg(y['time'], X, status=y['status']).fit()
    ref = sm.PHReg(S['time'], S[['x']], status=S['status']).fit()
    np.testing.assert_allclose(fit.params, ref.params)


def test_surv_errors(S):
    with pytest.raises(TypeError, match='by position or by keyword'):
        Surv('time', 'status', start='start')
    with pytest.raises(TypeError, match='takes'):
        Surv('time')
    with pytest.raises(ValueError, match='should be 0 or 1'):
        Surv('time', 'cases').fit(S)
    bad = S.assign(start=S['time'] + 1)
    with pytest.raises(ValueError, match='less than'):
        Surv('start', 'time', 'status').fit(bad)


# Binomial

def test_binomial_binary(Carseats):
    y = Binomial('US').fit_transform(Carseats)
    # the second level, as in R: No < Yes
    np.testing.assert_array_equal(y, (Carseats['US'] == 'Yes').astype(float))
    y_no = Binomial('US', level='No').fit_transform(Carseats)
    np.testing.assert_array_equal(y_no, 1 - y)


def test_binomial_binary_values(S):
    np.testing.assert_array_equal(Binomial('status').fit_transform(S), S['status'])
    np.testing.assert_array_equal(Binomial('flag').fit_transform(S), S['flag'].astype(float))
    with pytest.raises(ValueError, match='should be 0 or 1'):
        Binomial('cases').fit(S)
    with pytest.raises(ValueError, match='should have two levels'):
        Binomial('kind').fit(S)
    y = Binomial('kind', level='hi').fit_transform(S)
    np.testing.assert_array_equal(y, (S['kind'] == 'hi').astype(float))


def test_binomial_missing_and_unseen():
    D = pd.DataFrame({'x': [1., 2, 3, 4], 'y': ['a', 'b', None, 'a']})
    response = Binomial('y').fit(D)
    np.testing.assert_array_equal(response.transform(D), [0, 1, np.nan, 0])
    with pytest.raises(ValueError, match='not seen at fit'):
        response.transform(D.assign(y=['a', 'c', 'b', 'a']))


def test_binomial_counts_and_proportion_agree(S):
    X = MS(['x'], intercept=True).fit_transform(S)
    counts = Binomial('cases', trials='size').fit_transform(S)
    assert list(counts.columns) == ['cases', 'size-cases']
    np.testing.assert_array_equal(counts.sum(axis=1), S['size'])
    prop = Binomial('cases', trials='size', output='proportion').fit_transform(S)
    assert list(prop.columns) == ['cases/size', 'size']
    family = sm.families.Binomial()
    fit_counts = sm.GLM(counts, X, family=family).fit()
    fit_prop = sm.GLM(prop['cases/size'], X, family=family, var_weights=prop['size']).fit()
    np.testing.assert_allclose(fit_counts.params, fit_prop.params)
    with pytest.raises(ValueError, match='between 0'):
        Binomial('size', trials='cases').fit(S)
    with pytest.raises(ValueError, match="output should be"):
        Binomial('cases', trials='size', output='odds').fit(S)


# Multinomial

def test_multinomial_labels(S):
    response = Multinomial('kind').fit(S)
    y = response.transform(S)
    assert list(y.cat.categories) == ['lo', 'mid', 'hi']
    np.testing.assert_array_equal(y.astype(str), S['kind'].astype(str))
    with pytest.raises(ValueError, match='not seen at fit'):
        response.transform(S.assign(kind=S['kind'].cat.add_categories('new')
                                    .where(S.index != 0, 'new')))


def test_multinomial_onehot_and_counts(S):
    Y = Multinomial('kind', output='onehot').fit_transform(S)
    assert list(Y.columns) == ['kind[lo]', 'kind[mid]', 'kind[hi]']
    np.testing.assert_array_equal(Y.sum(axis=1), 1)
    np.testing.assert_array_equal(Y['kind[hi]'], (S['kind'] == 'hi').astype(float))
    C = Multinomial(['cases', 'size']).fit_transform(S)
    np.testing.assert_array_equal(C, S[['cases', 'size']].astype(float))
    strings = S.assign(kind=S['kind'].astype(str))
    y = Multinomial('kind').fit_transform(strings)
    assert list(y.cat.categories) == ['hi', 'lo', 'mid']   # sorted


def test_ndarray_output(S):
    A = np.asarray(S[['time', 'status', 'x']])
    y = Surv(0, 1).fit_transform(A)
    assert isinstance(y, np.ndarray) and y.shape == (40, 2)
    np.testing.assert_allclose(y, A[:, :2])


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_native_output(S, backend):
    lib = pytest.importorskip(backend)
    nw = pytest.importorskip('narwhals')
    P = lib.from_pandas(S) if backend == 'polars' else lib.Table.from_pandas(S)
    Y = Surv('time', 'status').fit_transform(P)
    assert nw.get_native_namespace(nw.from_native(Y)) is lib
    np.testing.assert_allclose(nw.from_native(Y).to_numpy(), S[['time', 'status']])
    y = Binomial('flag').fit_transform(P)
    assert nw.get_native_namespace(nw.from_native(y, series_only=True)) is lib
    np.testing.assert_allclose(np.asarray(y), S['flag'])
    labels = Multinomial('kind').fit_transform(P)
    assert nw.from_native(labels, series_only=True).to_list() == list(S['kind'])
    if backend == 'polars':
        # the levels of polars' (unordered) Categorical are sorted
        assert labels.dtype == lib.Enum(['hi', 'lo', 'mid'])
