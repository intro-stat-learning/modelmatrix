# The wildcard: all the columns of X but the response, as R's `.`

from itertools import combinations

import numpy as np, pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline

from modelmatrix import ModelSpec as MS, wildcard, Wildcard, pca, poly, derived_feature

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
    spec = MS([wildcard], response='Sales').fit(Carseats)
    _same(spec.transform(Carseats), MS(others).fit_transform(Carseats))
    assert spec.expanded_terms_ == others
    assert spec.names == others
    assert list(spec.terms) == [wildcard]   # terms are not modified


def test_no_response_all_columns(D):
    _same(MS([wildcard]).fit_transform(D), MS(list(D.columns)).fit_transform(D))


def test_explicit_terms_not_repeated(D):
    # R's y ~ x + .
    spec = MS(['x', wildcard], response='y').fit(D)
    assert spec.expanded_terms_ == ['x', 'z', 'g']
    spec = MS([wildcard, 'x', ('x', 'z')], response='y').fit(D)
    assert spec.expanded_terms_ == ['z', 'g', 'x', ('x', 'z')]
    # two wildcards: their columns once
    assert MS([wildcard, wildcard], response='y').fit(D).expanded_terms_ == ['x', 'z', 'g']


def test_pow(D):
    # R's y ~ .^2
    spec = MS([wildcard(pow=2)], response='y').fit(D)
    cols = ['x', 'z', 'g']
    assert spec.expanded_terms_ == cols + list(combinations(cols, 2))
    _same(spec.transform(D), MS(cols + list(combinations(cols, 2))).fit_transform(D))
    spec3 = MS([wildcard(pow=3)], response='y').fit(D)
    assert len(spec3.expanded_terms_) == 3 + 3 + 1
    # an interaction given explicitly, in another order, is not repeated
    spec = MS([('z', 'x'), wildcard(pow=2)], response='y').fit(D)
    assert spec.expanded_terms_ == [('z', 'x'), 'x', 'z', 'g', ('x', 'g'), ('z', 'g')]


def test_exclude(D):
    # R's y ~ . - z
    assert MS([wildcard(exclude='z')], response='y').fit(D).expanded_terms_ == ['x', 'g']
    assert MS([wildcard(exclude=['z', 'g'])], response='y').fit(D).expanded_terms_ == ['x']
    assert repr(wildcard(pow=2, exclude=['z'])) == '.^2 - z'
    assert repr(wildcard) == '.'


def test_interaction_with_wildcard(D):
    # R's y ~ .:g
    spec = MS([(wildcard, 'g')], response='y').fit(D)
    assert spec.expanded_terms_ == [('x', 'g'), ('z', 'g')]
    p = poly('x', 2)
    spec = MS([(p, wildcard(exclude='x'))], response='y').fit(D)
    assert spec.expanded_terms_ == [(p, 'z'), (p, 'g')]
    with pytest.raises(ValueError, match='cannot be a part of an interaction'):
        MS([(wildcard(pow=2), 'g')], response='y').fit(D)


def test_pca_of_wildcard(Carseats):
    numeric = [c for c in Carseats.columns.drop('Sales')
               if pd.api.types.is_numeric_dtype(Carseats[c])]
    pc = pca(wildcard(exclude=['ShelveLoc', 'Urban', 'US']), n_components=3)
    assert pc.name == 'pca(. - ShelveLoc - Urban - US, n_components=3)'
    spec = MS([pc], response='Sales').fit(Carseats)
    X = spec.transform(Carseats)
    X_explicit = MS([pca(numeric, name='pc', n_components=3)]).fit_transform(Carseats)
    np.testing.assert_allclose(X, X_explicit)
    assert X.shape == (400, 3)
    # the feature's variables are the columns, but the response
    assert spec.terms_[0].variables[:2] == ('CompPrice', 'Income')
    assert 'Sales' not in spec.terms_[0].variables


def test_wildcard_in_derived_feature(D):
    f = derived_feature(wildcard(exclude='g'), name='xz')
    X = MS([f], response='y').fit_transform(D)
    np.testing.assert_allclose(X, D[['x', 'z']])
    with pytest.raises(ValueError, match='cannot be a variable'):
        MS([derived_feature([wildcard(pow=2)], name='bad')]).fit(D)


def test_tuning_pca_of_wildcard(Carseats):
    pc = pca(wildcard(exclude=['ShelveLoc', 'Urban', 'US']), n_components=2)
    pipe = make_pipeline(MS([pc, 'ShelveLoc'], response='Sales'), LinearRegression())
    grid = GridSearchCV(pipe, {'modelspec__terms__0__n_components': [1, 3]}, cv=3)
    grid.fit(Carseats, Carseats['Sales'])
    best = grid.best_estimator_[0]
    assert best.terms_[0].encoder.n_components == grid.best_params_['modelspec__terms__0__n_components']


def test_transform_without_response(D):
    spec = MS([wildcard, ('x', 'g')], response='y').fit(D)
    _same(spec.transform(D.drop(columns='y')), spec.transform(D))


def test_clone_and_cross_validate(D):
    spec = MS([wildcard], response='y')
    cloned = clone(spec)
    assert cloned.terms == [wildcard]
    scores = cross_val_score(make_pipeline(spec, LinearRegression()), D, D['y'],
                             cv=3, error_score='raise')
    assert np.all(np.isfinite(scores))


def test_response_in_term_raises(D):
    with pytest.raises(ValueError, match='a response cannot be a term'):
        MS(['y', 'x'], response='y').fit(D)
    with pytest.raises(ValueError, match='a response cannot be a term'):
        MS([poly('y', 2)], response='y').fit(D)


def test_ndarray(D):
    A = np.asarray(D[['y', 'x', 'z']])
    spec = MS([wildcard], response=0).fit(A)
    assert spec.expanded_terms_ == [1, 2]
    np.testing.assert_allclose(spec.transform(A), A[:, 1:])
    np.testing.assert_allclose(spec.build_response(A), A[:, 0])
    # an explicit column is not repeated
    spec = MS([2, wildcard], response=0).fit(A)
    assert spec.expanded_terms_ == [2, 1]
    assert all(type(t) is int for t in spec.expanded_terms_)


def test_polars(D):
    pl = pytest.importorskip('polars')
    P = pl.from_pandas(D)
    spec = MS([wildcard], response='y').fit(P)
    X = spec.transform(P)
    assert isinstance(X, pl.DataFrame)
    np.testing.assert_allclose(X.to_numpy(), MS(['x', 'z', 'g']).fit_transform(D))
    y = spec.build_response(P)
    np.testing.assert_allclose(y, D['y'])


def test_wildcard_validation():
    with pytest.raises(ValueError, match='positive integer'):
        Wildcard(pow=0)
    with pytest.raises(ValueError, match='positive integer'):
        wildcard(pow=1.5)
    assert wildcard(pow=2) == Wildcard(pow=2)
    assert hash(wildcard(exclude=['x'])) == hash(Wildcard(exclude=('x',)))
