# The wildcard: all the columns of X (but those excluded), as R's `.`

from itertools import combinations

import numpy as np, pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline

from modelmatrix import (ModelSpec as MS, wildcard, Wildcard, pca, poly, derived_feature,
                         Surv, Binomial)

from _datasets import load_data


@pytest.fixture(scope='module')
def Carseats():
    return load_data('Carseats')


@pytest.fixture
def D():
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({'y': rng.standard_normal(n),
                         'x': rng.standard_normal(n),
                         'z': rng.uniform(size=n),
                         'g': pd.Categorical(rng.choice(['a', 'b', 'c'], n))})


def _same(A, B):
    assert list(A.columns) == list(B.columns)
    np.testing.assert_allclose(A, B)


def test_all_but_response(Carseats):
    others = list(Carseats.columns.drop('Sales'))
    spec = MS([wildcard(exclude='Sales')]).fit(Carseats)
    _same(spec.transform(Carseats), MS(others).fit_transform(Carseats))
    assert spec.expanded_terms_ == others
    assert spec.names == others
    assert list(spec.terms) == [wildcard(exclude='Sales')]   # terms are not modified


def test_all_columns(D):
    _same(MS([wildcard]).fit_transform(D), MS(list(D.columns)).fit_transform(D))


def test_explicit_terms_not_repeated(D):
    # R's y ~ x + .
    dot = wildcard(exclude='y')
    spec = MS(['x', dot]).fit(D)
    assert spec.expanded_terms_ == ['x', 'z', 'g']
    spec = MS([dot, 'x', ('x', 'z')]).fit(D)
    assert spec.expanded_terms_ == ['z', 'g', 'x', ('x', 'z')]
    # two wildcards: their columns once
    assert MS([dot, dot]).fit(D).expanded_terms_ == ['x', 'z', 'g']


def test_pow(D):
    # R's y ~ .^2
    spec = MS([wildcard(pow=2, exclude='y')]).fit(D)
    cols = ['x', 'z', 'g']
    assert spec.expanded_terms_ == cols + list(combinations(cols, 2))
    _same(spec.transform(D), MS(cols + list(combinations(cols, 2))).fit_transform(D))
    spec3 = MS([wildcard(pow=3, exclude='y')]).fit(D)
    assert len(spec3.expanded_terms_) == 3 + 3 + 1
    # an interaction given explicitly, in another order, is not repeated
    spec = MS([('z', 'x'), wildcard(pow=2, exclude='y')]).fit(D)
    assert spec.expanded_terms_ == [('z', 'x'), 'x', 'z', 'g', ('x', 'g'), ('z', 'g')]


def test_exclude(D):
    # R's y ~ . - z
    assert MS([wildcard(exclude='z')]).fit(D).expanded_terms_ == ['y', 'x', 'g']
    assert MS([wildcard(exclude=['y', 'z', 'g'])]).fit(D).expanded_terms_ == ['x']
    assert repr(wildcard(pow=2, exclude=['z'])) == '.^2 - z'
    assert repr(wildcard) == '.'


def test_interaction_with_wildcard(D):
    # R's y ~ .:g
    spec = MS([(wildcard(exclude='y'), 'g')]).fit(D)
    assert spec.expanded_terms_ == [('x', 'g'), ('z', 'g')]
    p = poly('x', 2)
    spec = MS([(p, wildcard(exclude=['y', 'x']))]).fit(D)
    assert spec.expanded_terms_ == [(p, 'z'), (p, 'g')]
    with pytest.raises(ValueError, match='cannot be a part of an interaction'):
        MS([(wildcard(pow=2), 'g')]).fit(D)


def test_pca_of_wildcard(Carseats):
    numeric = [c for c in Carseats.columns.drop('Sales')
               if pd.api.types.is_numeric_dtype(Carseats[c])]
    pc = pca(wildcard(exclude=['Sales', 'ShelveLoc', 'Urban', 'US']), n_components=3)
    assert pc.name == 'pca(. - Sales - ShelveLoc - Urban - US, n_components=3)'
    spec = MS([pc]).fit(Carseats)
    X = spec.transform(Carseats)
    X_explicit = MS([pca(numeric, name='pc', n_components=3)]).fit_transform(Carseats)
    np.testing.assert_allclose(X, X_explicit)
    assert X.shape == (400, 3)
    # the feature's variables are the columns, but those excluded
    assert spec.terms_[0].variables[:2] == ('CompPrice', 'Income')
    assert 'Sales' not in spec.terms_[0].variables


def test_wildcard_in_derived_feature(D):
    f = derived_feature(wildcard(exclude=['y', 'g']), name='xz')
    X = MS([f]).fit_transform(D)
    np.testing.assert_allclose(X, D[['x', 'z']])
    with pytest.raises(ValueError, match='cannot be a variable'):
        MS([derived_feature([wildcard(pow=2)], name='bad')]).fit(D)


def test_tuning_pca_of_wildcard(Carseats):
    pc = pca(wildcard(exclude=['Sales', 'ShelveLoc', 'Urban', 'US']), n_components=2)
    pipe = make_pipeline(MS([pc, 'ShelveLoc']), LinearRegression())
    grid = GridSearchCV(pipe, {'modelspec__terms__0__n_components': [1, 3]}, cv=3)
    grid.fit(Carseats, Carseats['Sales'])
    best = grid.best_estimator_[0]
    assert best.terms_[0].encoder.n_components == grid.best_params_['modelspec__terms__0__n_components']


def test_transform_without_response(D):
    spec = MS([wildcard(exclude='y'), ('x', 'g')]).fit(D)
    _same(spec.transform(D.drop(columns='y')), spec.transform(D))


def test_clone_and_cross_validate(D):
    spec = MS([wildcard(exclude='y')])
    cloned = clone(spec)
    assert cloned.terms == [wildcard(exclude='y')]
    scores = cross_val_score(make_pipeline(spec, LinearRegression()), D, D['y'],
                             cv=3, error_score='raise')
    assert np.all(np.isfinite(scores))


def test_exclude_response(D, Carseats):
    # a Response is excluded by its columns
    S = D.assign(time=np.arange(1, 31.), status=np.tile([0, 1], 15))
    surv = Surv('time', 'status')
    dot = wildcard(exclude=[surv, 'y'])
    assert dot == wildcard(exclude=['time', 'status', 'y'])
    assert repr(dot) == '. - time - status - y'
    assert MS([dot]).fit(S).expanded_terms_ == ['x', 'z', 'g']
    US = Binomial('US')
    spec = MS([wildcard(exclude=US)]).fit(Carseats)
    assert 'US' not in spec.expanded_terms_ and len(spec.expanded_terms_) == 10


def test_ndarray(D):
    A = np.asarray(D[['y', 'x', 'z']])
    spec = MS([wildcard(exclude=0)]).fit(A)
    assert spec.expanded_terms_ == [1, 2]
    np.testing.assert_allclose(spec.transform(A), A[:, 1:])
    # an explicit column is not repeated
    spec = MS([2, wildcard(exclude=0)]).fit(A)
    assert spec.expanded_terms_ == [2, 1]
    assert all(type(t) is int for t in spec.expanded_terms_)


def test_polars(D):
    pl = pytest.importorskip('polars')
    P = pl.from_pandas(D)
    spec = MS([wildcard(exclude='y')]).fit(P)
    X = spec.transform(P)
    assert isinstance(X, pl.DataFrame)
    np.testing.assert_allclose(X.to_numpy(), MS(['x', 'z', 'g']).fit_transform(D))


def test_wildcard_validation():
    with pytest.raises(ValueError, match='positive integer'):
        Wildcard(pow=0)
    with pytest.raises(ValueError, match='positive integer'):
        wildcard(pow=1.5)
    assert wildcard(pow=2) == Wildcard(pow=2)
    assert hash(wildcard(exclude=['x'])) == hash(Wildcard(exclude=('x',)))


def test_build_submodel_with_wildcard(D):
    spec = MS([wildcard(exclude='y')]).fit(D)
    _same(spec.build_submodel(D, [wildcard(exclude='y')]), spec.transform(D))
    _same(spec.build_submodel(D, ['x', wildcard(exclude=['y', 'g'])]),
          spec.build_submodel(D, ['x', 'z']))
