# Outcome: a response with an offset, weights, strata, ..., named by role

import numpy as np, pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer

from modelmatrix import (ModelSpec as MS, Outcome, Surv, Binomial, Multinomial,
                         wildcard, derived_feature, poly)


@pytest.fixture
def D():
    rng = np.random.default_rng(2)
    n = 60
    exposure = rng.uniform(1, 5, n)
    x = rng.standard_normal(n)
    return pd.DataFrame({'time': rng.exponential(size=n) + 0.1,
                         'status': rng.binomial(1, 0.7, n),
                         'O': rng.standard_normal(n) / 4,
                         'w': rng.uniform(0.5, 2, n),
                         'inst': pd.Categorical(rng.choice(['a', 'b', 'c'], n)),
                         'exposure': exposure,
                         'x': x,
                         'claims': rng.poisson(exposure * np.exp(0.3 * x)),
                         'cases': rng.binomial(8, 0.4, n),
                         'size': np.full(n, 8)})


log_exposure = derived_feature(['exposure'], encoder=FunctionTransformer(np.log),
                               name='log(exposure)')


def test_columns_named_by_role(D):
    Y = Outcome(Surv('time', 'status'), offset='O', weight='w', strata='inst')
    y = Y.fit_transform(D)
    assert list(y.columns) == ['stop', 'status', 'offset', 'weight', 'strata']
    np.testing.assert_allclose(y['offset'], D['O'])
    np.testing.assert_allclose(y['weight'], D['w'])
    assert isinstance(y['strata'].dtype, pd.CategoricalDtype)   # labels, not contrasts
    assert y.index.equals(D.index)
    assert repr(Y) == "Outcome(Surv('time', 'status'), offset='O', weight='w', strata='inst')"


def test_wildcard_excludes_every_column(D):
    Y = Outcome(Surv('time', 'status'), offset='O', weight='w', strata='inst')
    assert Y.columns == ('time', 'status', 'O', 'w', 'inst')
    spec = MS([wildcard(exclude=Y)]).fit(D)
    assert spec.expanded_terms_ == ['exposure', 'x', 'claims', 'cases', 'size']
    Z = Outcome('claims', offset=log_exposure)
    assert Z.columns == ('claims', 'exposure')
    assert MS([wildcard(exclude=Z)]).fit(D).expanded_terms_ == [
        'time', 'status', 'O', 'w', 'inst', 'x', 'cases', 'size']


def test_poisson_offset_matches_statsmodels(D):
    # R's glm(claims ~ x + offset(log(exposure)), family=poisson)
    Y = Outcome('claims', offset=log_exposure)
    y = Y.fit_transform(D)
    assert list(y.columns) == ['claims', 'offset']
    X = MS(['x'], intercept=True).fit_transform(D)
    family = sm.families.Poisson()
    fit = sm.GLM(y['claims'], X, family=family, offset=y['offset']).fit()
    ref = sm.GLM(D['claims'], X, family=family, offset=np.log(D['exposure'])).fit()
    np.testing.assert_allclose(fit.params, ref.params)


def test_offset_of_new_data(D):
    Y = Outcome('claims', offset=log_exposure, weight='w').fit(D)
    new = D.drop(columns='claims').iloc[:5]
    O = Y.transform(new, response=False)
    assert list(O.columns) == ['offset', 'weight']
    np.testing.assert_allclose(O['offset'], np.log(new['exposure']))
    with pytest.raises(ValueError, match="'claims' is not in X"):
        Y.transform(new)


def test_stratified_cox_matches_statsmodels(D):
    Y = Outcome(Surv('time', 'status'), offset='O', strata='inst')
    y = Y.fit_transform(D)
    X = MS(['x']).fit_transform(D)
    fit = sm.PHReg(y['stop'], X, status=y['status'], strata=y['strata'],
                   offset=y['offset']).fit()
    ref = sm.PHReg(D['time'], D[['x']], status=D['status'], strata=D['inst'],
                   offset=D['O']).fit()
    np.testing.assert_allclose(fit.params, ref.params)


def test_binomial_proportion_weights(D):
    # the trials are the weights of a proportion, times the prior weights
    y = Outcome(Binomial('cases', trials='size', output='proportion')).fit_transform(D)
    assert list(y.columns) == ['cases/size', 'weight']
    np.testing.assert_allclose(y['weight'], D['size'])
    y = Outcome(Binomial('cases', trials='size', output='proportion'), weight='w').fit_transform(D)
    assert list(y.columns) == ['cases/size', 'weight']
    np.testing.assert_allclose(y['weight'], D['size'] * D['w'])
    # counts are left as they are
    y = Outcome(Binomial('cases', trials='size'), weight='w').fit_transform(D)
    assert list(y.columns) == ['cases', 'size-cases', 'weight']


def test_other_responses(D):
    y = Outcome(Multinomial('inst'), weight='w').fit_transform(D)
    assert list(y.columns) == ['inst', 'weight']
    assert isinstance(y['inst'].dtype, pd.CategoricalDtype)
    # the response given is not modified by fit
    response = Binomial('status')
    Outcome(response, weight='w').fit(D)
    assert not hasattr(response, 'levels_')


def test_errors(D):
    with pytest.raises(NotFittedError):
        Outcome('claims', offset='O').transform(D)
    with pytest.raises(ValueError, match='single column'):
        Outcome('claims', offset=poly('x', 2)).fit(D)
    with pytest.raises(ValueError, match='same names'):
        Outcome('claims', claims='O').fit(D)
    with pytest.raises(ValueError, match="'nope' is not in X"):
        Outcome('claims', offset='nope').fit(D)
    with pytest.raises(ValueError, match='wildcard'):
        Outcome('claims', offset=derived_feature(wildcard, name='all')).columns


def test_ndarray(D):
    A = np.asarray(D[['time', 'status', 'O']], dtype=float)
    y = Outcome(Surv(0, 1), offset=2).fit_transform(A)
    assert isinstance(y, np.ndarray)
    np.testing.assert_allclose(y, A)


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_native(D, backend):
    lib = pytest.importorskip(backend)
    nw = pytest.importorskip('narwhals')
    P = lib.from_pandas(D) if backend == 'polars' else lib.Table.from_pandas(D)
    y = Outcome(Surv('time', 'status'), offset='O', strata='inst').fit_transform(P)
    y_nw = nw.from_native(y)
    assert nw.get_native_namespace(y_nw) is lib
    assert y_nw.columns == ['stop', 'status', 'offset', 'strata']
    assert y_nw['strata'].to_list() == list(D['inst'])
    if backend == 'polars':
        assert y['strata'].dtype == lib.Enum(['a', 'b', 'c'])


# multi-response models: one column for each response

@pytest.fixture
def M():
    rng = np.random.default_rng(3)
    n = 30
    return pd.DataFrame({'y1': rng.standard_normal(n), 'y2': rng.standard_normal(n),
                         'O1': rng.standard_normal(n), 'O2': rng.standard_normal(n),
                         'O': rng.standard_normal(n), 'exposure': rng.uniform(1, 5, n),
                         'kind': pd.Categorical(rng.choice(['lo', 'mid', 'hi'], n),
                                                categories=['lo', 'mid', 'hi']),
                         'x': rng.standard_normal(n)})


def test_offset_for_each_response(M):
    # in the order of the responses, whatever the order of the dict
    Y = Outcome(['y1', 'y2'], offset={'y2': 'O2', 'y1': log_exposure})
    y = Y.fit_transform(M)
    assert list(y.columns) == ['y1', 'y2', 'offset[y1]', 'offset[y2]']
    np.testing.assert_allclose(y[['y1', 'y2']], M[['y1', 'y2']])
    np.testing.assert_allclose(y['offset[y1]'], np.log(M['exposure']))
    np.testing.assert_allclose(y['offset[y2]'], M['O2'])
    assert Y.columns == ('y1', 'y2', 'O2', 'exposure')
    assert MS([wildcard(exclude=Y)]).fit(M).expanded_terms_ == ['O1', 'O', 'kind', 'x']
    assert repr(Y) == "Outcome(['y1', 'y2'], offset={'y2': 'O2', 'y1': log(exposure)})"
    # new data, without the responses
    O = Y.transform(M.drop(columns=['y1', 'y2']), response=False)
    assert list(O.columns) == ['offset[y1]', 'offset[y2]']


def test_one_offset_for_every_response(M):
    y = Outcome(['y1', 'y2'], offset='O').fit_transform(M)
    assert list(y.columns) == ['y1', 'y2', 'offset']


@pytest.mark.parametrize('output', ['labels', 'onehot'])
def test_offset_for_each_class(M, output):
    # keyed by the classes, whatever the output of the Multinomial
    Y = Outcome(Multinomial('kind', output=output),
                offset={'hi': 'O', 'lo': 'O1', 'mid': 'O2'})
    y = Y.fit_transform(M)
    offsets = ['offset[lo]', 'offset[mid]', 'offset[hi]']
    assert list(y.columns)[-3:] == offsets
    np.testing.assert_allclose(y[offsets], M[['O1', 'O2', 'O']])
    # the classes of counts are their columns
    C = M.assign(n_lo=1, n_hi=2)
    y = Outcome(Multinomial(['n_lo', 'n_hi']), offset={'n_lo': 'O1', 'n_hi': 'O2'}).fit_transform(C)
    assert list(y.columns) == ['n_lo', 'n_hi', 'offset[n_lo]', 'offset[n_hi]']


def test_offset_for_each_response_errors(M):
    with pytest.raises(ValueError, match=r"missing \['y2'\], unknown \['y3'\]"):
        Outcome(['y1', 'y2'], offset={'y1': 'O1', 'y3': 'O2'}).fit(M)
    with pytest.raises(ValueError, match="missing \\['hi'\\]"):
        Outcome(Multinomial('kind'), offset={'lo': 'O1', 'mid': 'O2'}).fit(M)
    with pytest.raises(ValueError, match='not a multi-response'):
        Outcome('y1', offset={'y1': 'O1'}).fit(M)
    with pytest.raises(ValueError, match='single column'):
        Outcome(['y1', 'y2'], offset={'y1': poly('x', 2), 'y2': 'O2'}).fit(M)


def test_offset_for_each_response_polars(M):
    pl = pytest.importorskip('polars')
    y = Outcome(['y1', 'y2'], offset={'y1': 'O1', 'y2': 'O2'}).fit_transform(pl.from_pandas(M))
    assert isinstance(y, pl.DataFrame)
    assert y.columns == ['y1', 'y2', 'offset[y1]', 'offset[y2]']


def test_weights_for_each_response(M):
    W = M.assign(w1=1.0, w2=0.5)
    y = Outcome(['y1', 'y2'], offset='O', weight={'y1': 'w1', 'y2': 'w2'}).fit_transform(W)
    assert list(y.columns) == ['y1', 'y2', 'offset', 'weight[y1]', 'weight[y2]']
    np.testing.assert_allclose(y['weight[y2]'], 0.5)
