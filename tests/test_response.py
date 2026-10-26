# Responses: ModelSpec(..., response=...) and build_response

import numpy as np, pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.base import clone
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


# columns

def test_column(Carseats):
    spec = MS(['Price'], response='Sales').fit(Carseats)
    y = spec.build_response(Carseats)
    pd.testing.assert_series_equal(y, Carseats['Sales'])
    Y = MS(['Price'], response=['Sales', 'Income']).fit(Carseats).build_response(Carseats)
    pd.testing.assert_frame_equal(Y, Carseats[['Sales', 'Income']])


def test_no_response_or_not_fit(Carseats):
    with pytest.raises(NotFittedError):
        MS(['Price'], response='Sales').build_response(Carseats)
    with pytest.raises(ValueError, match='has no response'):
        MS(['Price']).fit(Carseats).build_response(Carseats)
    with pytest.raises(ValueError, match="'nope' is not in X"):
        MS(['Price'], response='nope').fit(Carseats)


def test_pipeline_with_response(Carseats):
    spec = MS([wildcard], response='Sales')
    y = clone(spec).fit(Carseats).build_response(Carseats)
    scores = cross_val_score(make_pipeline(spec, LinearRegression()), Carseats, y,
                             cv=3, error_score='raise')
    assert np.all(np.isfinite(scores))
    # the response object is not modified by fit
    surv = Surv('a', 'b')
    MS(['x'], response=surv)
    assert not hasattr(surv, 'levels_')


def test_index_of_y(Carseats):
    spec = MS(['Price'])
    spec.fit(Carseats, Carseats['Sales'])
    with pytest.raises(ValueError, match='indices of X and y do not agree'):
        spec.fit(Carseats, Carseats['Sales'].reset_index(drop=True).iloc[::-1])
    # arrays are not checked
    spec.fit(Carseats, np.asarray(Carseats['Sales']))


# Surv

def test_surv(S):
    y = MS(['x'], response=Surv('time', 'status')).fit(S).build_response(S)
    pd.testing.assert_frame_equal(y, S[['time', 'status']].astype(float))
    y3 = MS(['x'], response=Surv('start', 'time', 'status')).fit(S).build_response(S)
    assert list(y3.columns) == ['start', 'time', 'status']
    y_kw = MS(['x'], response=Surv(stop='time', status='status', start='start')).fit(S).build_response(S)
    pd.testing.assert_frame_equal(y3, y_kw)
    assert repr(Surv('start', 'time', 'status')) == "Surv('start', 'time', 'status')"
    # a boolean status
    yb = MS(['x'], response=Surv('time', 'flag')).fit(S).build_response(S)
    np.testing.assert_array_equal(yb['flag'], S['flag'].astype(float))


def test_surv_wildcard_excludes_its_columns(S):
    spec = MS([wildcard(exclude=['cases', 'size', 'flag', 'kind'])],
              response=Surv('start', 'time', 'status')).fit(S)
    assert spec.expanded_terms_ == ['x']


def test_surv_matches_statsmodels_PHReg(S):
    spec = MS(['x'], response=Surv('time', 'status')).fit(S)
    X, y = spec.transform(S), spec.build_response(S)
    fit = sm.PHReg(y['time'], X, status=y['status']).fit()
    ref = sm.PHReg(S['time'], S[['x']], status=S['status']).fit()
    np.testing.assert_allclose(fit.params, ref.params)


def test_surv_errors(S):
    with pytest.raises(TypeError, match='by position or by keyword'):
        Surv('time', 'status', start='start')
    with pytest.raises(TypeError, match='takes'):
        Surv('time')
    with pytest.raises(ValueError, match='should be 0 or 1'):
        MS(['x'], response=Surv('time', 'cases')).fit(S)
    bad = S.assign(start=S['time'] + 1)
    with pytest.raises(ValueError, match='less than'):
        MS(['x'], response=Surv('start', 'time', 'status')).fit(bad)


# Binomial

def test_binomial_binary(Carseats):
    y = MS(['Price'], response=Binomial('US')).fit(Carseats).build_response(Carseats)
    # the second level, as in R: No < Yes
    np.testing.assert_array_equal(y, (Carseats['US'] == 'Yes').astype(float))
    y_no = MS(['Price'], response=Binomial('US', level='No')).fit(Carseats).build_response(Carseats)
    np.testing.assert_array_equal(y_no, 1 - y)


def test_binomial_binary_values(S):
    y = MS(['x'], response=Binomial('status')).fit(S).build_response(S)
    np.testing.assert_array_equal(y, S['status'])
    y = MS(['x'], response=Binomial('flag')).fit(S).build_response(S)
    np.testing.assert_array_equal(y, S['flag'].astype(float))
    with pytest.raises(ValueError, match='should be 0 or 1'):
        MS(['x'], response=Binomial('cases')).fit(S)
    with pytest.raises(ValueError, match='should have two levels'):
        MS(['x'], response=Binomial('kind')).fit(S)
    y = MS(['x'], response=Binomial('kind', level='hi')).fit(S).build_response(S)
    np.testing.assert_array_equal(y, (S['kind'] == 'hi').astype(float))


def test_binomial_missing_and_unseen():
    D = pd.DataFrame({'x': [1., 2, 3, 4], 'y': ['a', 'b', None, 'a']})
    spec = MS(['x'], response=Binomial('y')).fit(D)
    y = spec.build_response(D)
    np.testing.assert_array_equal(y, [0, 1, np.nan, 0])
    with pytest.raises(ValueError, match='not seen at fit'):
        spec.build_response(D.assign(y=['a', 'c', 'b', 'a']))


def test_binomial_counts_and_proportion_agree(S):
    X = MS(['x'], intercept=True).fit_transform(S)
    counts = MS(['x'], response=Binomial('cases', trials='size')).fit(S).build_response(S)
    assert list(counts.columns) == ['cases', 'size-cases']
    np.testing.assert_array_equal(counts.sum(axis=1), S['size'])
    prop = MS(['x'], response=Binomial('cases', trials='size', output='proportion')).fit(S).build_response(S)
    assert list(prop.columns) == ['cases/size', 'size']
    family = sm.families.Binomial()
    fit_counts = sm.GLM(counts, X, family=family).fit()
    fit_prop = sm.GLM(prop['cases/size'], X, family=family, var_weights=prop['size']).fit()
    np.testing.assert_allclose(fit_counts.params, fit_prop.params)
    with pytest.raises(ValueError, match='between 0'):
        MS(['x'], response=Binomial('size', trials='cases')).fit(S)
    with pytest.raises(ValueError, match="output should be"):
        MS(['x'], response=Binomial('cases', trials='size', output='odds')).fit(S)


# Multinomial

def test_multinomial_labels(S):
    spec = MS(['x'], response=Multinomial('kind')).fit(S)
    y = spec.build_response(S)
    assert list(y.cat.categories) == ['lo', 'mid', 'hi']
    np.testing.assert_array_equal(y.astype(str), S['kind'].astype(str))
    with pytest.raises(ValueError, match='not seen at fit'):
        spec.build_response(S.assign(kind=S['kind'].cat.add_categories('new').fillna('new')
                                     .where(S.index != 0, 'new')))


def test_multinomial_onehot_and_counts(S):
    Y = MS(['x'], response=Multinomial('kind', output='onehot')).fit(S).build_response(S)
    assert list(Y.columns) == ['kind[lo]', 'kind[mid]', 'kind[hi]']
    np.testing.assert_array_equal(Y.sum(axis=1), 1)
    np.testing.assert_array_equal(Y['kind[hi]'], (S['kind'] == 'hi').astype(float))
    C = MS(['x'], response=Multinomial(['cases', 'size'])).fit(S).build_response(S)
    np.testing.assert_array_equal(C, S[['cases', 'size']].astype(float))
    strings = S.assign(kind=S['kind'].astype(str))
    y = MS(['x'], response=Multinomial('kind')).fit(strings).build_response(strings)
    assert list(y.cat.categories) == ['hi', 'lo', 'mid']   # sorted


def test_ndarray_output(S):
    A = np.asarray(S[['time', 'status', 'x']])
    y = MS([2], response=Surv(0, 1)).fit(A).build_response(A)
    assert isinstance(y, np.ndarray) and y.shape == (40, 2)
    np.testing.assert_allclose(y, A[:, :2])
