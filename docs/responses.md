---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Responses and the wildcard

As in scikit-learn, the design `X` and the response `y` are built
separately: a `ModelSpec` builds the design, and a response, such as
`Surv`, `Binomial` or `Multinomial`, builds `y` from the same data.
`wildcard` stands for every column, as `.` in R's formulae, and
`wildcard(exclude=...)` leaves out the response:
`ModelSpec([wildcard(exclude='Sales')])` is the design of R's
`Sales ~ .`.

```{code-cell} ipython3
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from ISLP import load_data
from modelmatrix import (ModelSpec as MS,
                         wildcard,
                         pca,
                         poly,
                         Surv,
                         Binomial,
                         Multinomial)

Carseats = load_data('Carseats')
```

## `y ~ .`

```{code-cell} ipython3
spec = MS([wildcard(exclude='Sales')], intercept=True).fit(Carseats)
X, y = spec.transform(Carseats), Carseats['Sales']
sm.OLS(y, X).fit().params
```

The wildcard is expanded when the `ModelSpec` is fit: `terms` is
unchanged, and `expanded_terms_` holds what it stood for.

```{code-cell} ipython3
spec.terms, spec.expanded_terms_
```

Columns that are terms themselves are not repeated, as in R's
`Sales ~ Price + .`, and more columns can be excluded, as R's `. - x`:

```{code-cell} ipython3
MS([poly('Price', 2), 'Price',
    wildcard(exclude=['Sales', 'Population', 'Education'])]).fit(Carseats).expanded_terms_
```

## `y ~ .^2` and `.:x`

`wildcard(pow=2)` adds every interaction of two columns, as R's `.^2`,
and the wildcard can be a part of an interaction, as R's `.:US`:

```{code-cell} ipython3
small = Carseats[['Sales', 'Price', 'Income', 'US']]
MS([wildcard(pow=2, exclude='Sales')]).fit(small).expanded_terms_
```

```{code-cell} ipython3
MS(['US', (wildcard(exclude=['Sales', 'US']), 'US')]).fit(small).expanded_terms_
```

## Features of the wildcard

A wildcard can be a variable of a feature: here, the first principal
components of the numeric columns, but the response.

```{code-cell} ipython3
pc = pca(wildcard(exclude=['Sales', 'ShelveLoc', 'Urban', 'US']), n_components=3, scale=True)
spec = MS([pc, 'ShelveLoc']).fit(Carseats)
spec.transform(Carseats).iloc[:5]
```

Its parameters are tuned as those of any term:

```{code-cell} ipython3
pipe = make_pipeline(MS([pc, 'ShelveLoc']), LinearRegression())
grid = GridSearchCV(pipe, {'modelspec__terms__0__n_components': [1, 3, 5, 7]}, cv=5)
grid.fit(Carseats, Carseats['Sales'])
grid.best_params_
```

When `X` and `y` are both pandas objects, `ModelSpec.fit` checks that
their indices agree.

## Responses

A response is fit, e.g. to learn the levels of a categorical, so that
it is coded the same way on new data, and `transform` builds it:
`response.fit_transform(D)`. For a dataframe of another backend, such
as polars, the response is in that backend. `wildcard(exclude=response)`
leaves out the response's columns, so that it is not a term of its own
design.

### Survival responses: `Surv`

`Surv(time, status)` is R's survival response, and `Surv(start, stop,
status)` that of (start, stop] intervals. The response is a dataframe
whose columns are named by their role, whatever their names in the
data: `stop` (the times), `status` and, if given, `start`, as R's
`Surv` and lifelines' `CoxTimeVaryingFitter` name them.

```{code-cell} ipython3
BrainCancer = load_data('BrainCancer').dropna()
response = Surv('time', 'status')
y = response.fit_transform(BrainCancer)
X = MS([wildcard(exclude=response)]).fit_transform(BrainCancer)
y.iloc[:5]
```

```{code-cell} ipython3
sm.PHReg(y['stop'], X, status=y['status']).fit().params
```

### Binomial responses

`Binomial('US')` codes a binary column as 0 and 1, the second level, as in
R, being a success (or the one given by `level=`):

```{code-cell} ipython3
y = Binomial('US').fit_transform(Carseats)
pd.crosstab(y, Carseats['US'])
```

With the number of trials, `Binomial(successes, trials=...)` gives the
successes and the failures, as R's `cbind(successes, failures)` (which
statsmodels' `GLM` takes too), or with `output='proportion'` the proportion
and the number of trials, as a response and weights:

```{code-cell} ipython3
rng = np.random.default_rng(0)
D = pd.DataFrame({'x': rng.standard_normal(50), 'size': rng.integers(5, 20, 50)})
D['cases'] = rng.binomial(D['size'], 1 / (1 + np.exp(-D['x'])))

counts = Binomial('cases', trials='size').fit_transform(D)
prop = Binomial('cases', trials='size', output='proportion').fit_transform(D)
X = MS(['x'], intercept=True).fit_transform(D)
family = sm.families.Binomial()
(sm.GLM(counts, X, family=family).fit().params,
 sm.GLM(prop['cases/size'], X, family=family, var_weights=prop['size']).fit().params)
```

### Multinomial responses

`Multinomial('ShelveLoc')` gives the class of each row as a categorical
whose categories are the levels seen at fit, and with `output='onehot'`
an indicator column for each level; `Multinomial([...])` takes a column
of counts for each class.

```{code-cell} ipython3
Multinomial('ShelveLoc', output='onehot').fit_transform(Carseats).iloc[:5]
```

### Offsets, weights and strata: `Outcome`

`Outcome` puts the other columns a model takes with its response next
to it, each named by its keyword: an offset, weights, the strata of a
Cox model, or any other. A column is taken as it is (a categorical keeps
its labels), and a feature of a single column is fit as a term of a
`ModelSpec`, e.g. R's `offset(log(exposure))`:

```{code-cell} ipython3
from sklearn.preprocessing import FunctionTransformer
from modelmatrix import Outcome, derived_feature

rng = np.random.default_rng(1)
P = pd.DataFrame({'x': rng.standard_normal(100), 'exposure': rng.uniform(1, 5, 100)})
P['claims'] = rng.poisson(P['exposure'] * np.exp(0.3 * P['x']))

log_exposure = derived_feature(['exposure'], encoder=FunctionTransformer(np.log),
                               name='log(exposure)')
outcome = Outcome('claims', offset=log_exposure)
y = outcome.fit_transform(P)
X = MS([wildcard(exclude=outcome)], intercept=True).fit_transform(P)   # leaves out exposure too
sm.GLM(y['claims'], X, family=sm.families.Poisson(), offset=y['offset']).fit().params
```

As the columns are named by their role, an estimator can find them by
name, e.g. a stratified Cox model with an offset:

```{code-cell} ipython3
BrainCancer['O'] = 0.1
outcome = Outcome(Surv('time', 'status'), offset='O', strata='sex')
y = outcome.fit_transform(BrainCancer)
y.iloc[:5]
```

The offset of new data, whose response is not known, is
`outcome.transform(new, response=False)`. With a `Binomial` of
`output='proportion'`, the number of trials is the column `weight`
(times the weights given as `weight=`), as R's weights of a proportion.

## Missing values

As scikit-learn's transformers, a `ModelSpec` keeps every row: a row
of the design is missing where a variable it uses is missing (a
`poly` or spline is fit on the observed values), and so is the
response. A transformer cannot drop rows of `X` without those of `y`,
so the rows to keep, those of R's `na.omit`, are chosen outside the
`ModelSpec` (or the missing values are imputed, e.g. with
scikit-learn's `SimpleImputer`, for an estimator that does not take
them).

```{code-cell} ipython3
Hitters = load_data('Hitters')
X = MS([wildcard(exclude='Salary')]).fit_transform(Hitters)
y = Hitters['Salary']
keep = X.notna().all(axis=1) & y.notna()
X, y = X[keep], y[keep]
X.shape
```
