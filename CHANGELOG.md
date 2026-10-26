# Changelog

## Unreleased

### New features

- **Responses:** `ModelSpec(terms, response=...)` and
  `spec.build_response(X)`, the response built from the same data as the
  design. A response is a column, a list of columns, or
  - `Surv(time, status)` or `Surv(start, stop, status)`, as R's: a
    `pd.DataFrame` of these columns, as glmstar's `CoxFamily` or lifelines
    take it;
  - `Binomial('y')` for a binary column (0/1, boolean or two levels; the
    second level, or `level=`, is a success), or `Binomial(successes,
    trials=...)` for the successes and failures (`output='counts'`, R's
    `cbind`) or the proportion and number of trials (`output='proportion'`);
  - `Multinomial('class')` for the class of each row, as a categorical or
    one-hot (`output='onehot'`), or `Multinomial([...])` for counts.

  Responses learn their levels at fit; a level unseen at fit raises.
  The response's columns cannot be in a term.
- **`wildcard`**, R's `.`: every column of `X` but the response, so
  `ModelSpec([wildcard], response='y')` is `y ~ .`. `wildcard(pow=2)` is
  `.^2`, `wildcard(exclude=['x'])` is `. - x`, `(wildcard, 'g')` is `.:g`,
  and `pca(wildcard, n_components=3)` a feature of every column. It is
  expanded at fit into `expanded_terms_`; columns that are terms already
  are not repeated, as in R's `y ~ x + .`.
- `pca()`'s `name` is optional.
- `fit(X, y)` checks that the indices of `X` and `y` agree when both are
  pandas objects.

### Documentation

- A page on responses and the wildcard.

## 0.1.0 (2026-10-26)

`ModelSpec`, from [ISLP](https://github.com/intro-stat-learning/ISLP) 0.6.0,
as its own package, with the history of its files. ISLP 0.7 depends on it,
and `from ISLP.models import ModelSpec` keeps working there.

### Changes from ISLP 0.6.0

- **`intercept` defaults to `False`:** pass `intercept=True` for a column
  of ones named `'intercept'`, as statsmodels' models (`OLS`, `GLM`, ...)
  need. Most scikit-learn estimators fit the intercept themselves.
- No `FutureWarning` of the move.
- Import from `modelmatrix` (`ModelSpec`, `Column`, `Feature`, `Contrast`,
  `Ordinal`, `poly`, `ns`, `bs`, `pca`, `contrast`, `ordinal`,
  `derived_feature`, `build_columns`); the transforms are in
  `modelmatrix.transforms`.
- The docs: the guide to `ModelSpec`, the pages on polynomials, splines,
  PCA and ANOVA, and the comparison with `ColumnTransformer`, patsy and
  formulaic, from ISLP's docs, as a Jupyter Book 2 site (as glmstar's),
  with a Sphinx API reference.

For the changes up to ISLP 0.6.0, see
[ISLP's changelog](https://github.com/intro-stat-learning/ISLP/blob/main/CHANGELOG.md).

Requires Python 3.11 or later.
