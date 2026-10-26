# Changelog

## Unreleased

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
