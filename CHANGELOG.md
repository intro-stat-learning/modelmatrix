# Changelog

## Unreleased

### New features

- **Offsets of multi-response models**: in `Outcome`, a dict gives one
  column for each response, keyed by the responses: the columns of a
  list of columns (now a response of `Outcome` too), or the classes of a
  `Multinomial`, whatever its output.
  `Outcome(['y1', 'y2'], offset={'y1': 'O1', 'y2': 'O2'})` gives the
  columns `y1`, `y2`, `offset[y1]` and `offset[y2]`, in the order of the
  responses; a missing or unknown key raises. A single column,
  `offset='O'`, is still one column, the same for every response.

## 0.3.0 (2026-10-26)

### New features

- **`Outcome`**: a response with the other columns a model takes with
  it, each named by its keyword, so an estimator can find them by name:
  `Outcome(Surv('time', 'status'), offset='O', weight='w', strata='inst')`
  gives the columns `stop`, `status`, `offset`, `weight` and `strata`. A
  column is taken as it is (a categorical keeps its labels); a feature of
  a single column is fit as a term, e.g. `offset(log(exposure))`.
  `wildcard(exclude=outcome)` leaves out all of its columns, and
  `transform(X, response=False)` builds the offset of new data. With
  `Binomial(..., output='proportion')`, the trials are the column
  `weight`, times any weights given.
- Dataframes of other backends keep a categorical column (as a polars
  `Enum`).

### Breaking changes

- `Surv` names its columns by their role, `start`, `stop` and `status`
  (as R's and lifelines'), not as in the data: `Surv('time', 'status')`
  gives the columns `stop` and `status`.

## 0.2.0 (2026-10-26)

### New features

- **Responses**, built from the same data as the design, as a
  `ModelSpec` builds it: `response.fit(D)`, then `response.transform(D)`
  (or `fit_transform`), in the backend of `D` (e.g. polars).
  - `Surv(time, status)` or `Surv(start, stop, status)`, as R's: a
    dataframe of these columns, as glmstar's `CoxFamily` or lifelines
    take it;
  - `Binomial('y')` for a binary column (0/1, boolean or two levels; the
    second level, or `level=`, is a success), or `Binomial(successes,
    trials=...)` for the successes and failures (`output='counts'`, R's
    `cbind`) or the proportion and number of trials (`output='proportion'`);
  - `Multinomial('class')` for the class of each row, as a categorical or
    one-hot (`output='onehot'`), or `Multinomial([...])` for counts.

  Responses learn their levels at fit; a level unseen at fit raises.
- **`wildcard`**, R's `.`: every column of `X`, but those of
  `wildcard(exclude=...)`, so `ModelSpec([wildcard(exclude='y')])` is the
  design of `y ~ .`. `exclude` also takes a response, leaving out its
  columns: `wildcard(exclude=Surv('time', 'status'))`. `wildcard(pow=2)`
  is `.^2`, `(wildcard, 'g')` is `.:g`, and `pca(wildcard(exclude='y'),
  n_components=3)` a feature of the columns. It is expanded at fit into
  `expanded_terms_`; columns that are terms already are not repeated, as
  in R's `y ~ x + .`. `build_submodel` expands it too.
- `pca()`'s `name` is optional.
- `fit(X, y)` checks that the indices of `X` and `y` agree when both are
  pandas objects.

### Bug fixes

- `poly`, `ns` and `bs` are fit on the observed values of their variable,
  and are missing only in its missing rows: before, a single missing value
  made every value of the feature missing (or, for `ns`, arbitrary).

### Documentation

- A page on responses and the wildcard, and on missing values.

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
