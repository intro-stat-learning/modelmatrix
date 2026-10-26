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

A `ModelSpec` can also describe the response of a model,
`ModelSpec(terms, response=...)`, and `build_response` then builds it
from the same data as the design. With a response,
`wildcard` stands for every other column, as `.` in R's
formulae: `ModelSpec([wildcard], response='Sales')` is
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
spec = MS([wildcard], response='Sales', intercept=True).fit(Carseats)
X, y = spec.transform(Carseats), spec.build_response(Carseats)
sm.OLS(y, X).fit().params
```

The wildcard is expanded when the `ModelSpec` is fit: `terms` is
unchanged, and `expanded_terms_` holds what it stood for.

```{code-cell} ipython3
spec.terms, spec.expanded_terms_
```

Columns that are terms themselves are not repeated, as in R's
`Sales ~ Price + .`, and `wildcard(exclude=...)` is R's `. - x`:

```{code-cell} ipython3
MS([poly('Price', 2), 'Price', wildcard(exclude=['Population', 'Education'])],
   response='Sales').fit(Carseats).expanded_terms_
```

## `y ~ .^2` and `.:x`

`wildcard(pow=2)` adds every interaction of two columns, as R's `.^2`,
and the wildcard can be a part of an interaction, as R's `.:US`:

```{code-cell} ipython3
small = Carseats[['Sales', 'Price', 'Income', 'US']]
MS([wildcard(pow=2)], response='Sales').fit(small).expanded_terms_
```

```{code-cell} ipython3
MS(['US', (wildcard, 'US')], response='Sales').fit(small).expanded_terms_
```

## Features of the wildcard

A wildcard can be a variable of a feature: here, the first principal
components of the numeric columns, but the response.

```{code-cell} ipython3
pc = pca(wildcard(exclude=['ShelveLoc', 'Urban', 'US']), n_components=3, scale=True)
spec = MS([pc, 'ShelveLoc'], response='Sales').fit(Carseats)
spec.transform(Carseats).iloc[:5]
```

Its parameters are tuned as those of any term:

```{code-cell} ipython3
y = spec.build_response(Carseats)
pipe = make_pipeline(MS([pc, 'ShelveLoc'], response='Sales'), LinearRegression())
grid = GridSearchCV(pipe, {'modelspec__terms__0__n_components': [1, 3, 5, 7]}, cv=5)
grid.fit(Carseats, y)
grid.best_params_
```

The response is passed to `fit` as `y` (a `Pipeline` passes the design's
data only to the `ModelSpec`). When both are pandas objects, `ModelSpec.fit`
checks that their indices agree.

## Survival responses: `Surv`

`Surv(time, status)` is R's survival response, and `Surv(start, stop,
status)` that of (start, stop] intervals. The response is a
`pd.DataFrame` with these columns, named as in the data, as glmstar's
`CoxFamily(event_id=..., status_id=..., start_id=...)` or lifelines take it.

```{code-cell} ipython3
BrainCancer = load_data('BrainCancer').dropna()
spec = MS([wildcard], response=Surv('time', 'status')).fit(BrainCancer)
X, y = spec.transform(BrainCancer), spec.build_response(BrainCancer)
y.iloc[:5]
```

```{code-cell} ipython3
sm.PHReg(y['time'], X, status=y['status']).fit().params
```

## Binomial responses

`Binomial('US')` codes a binary column as 0 and 1, the second level, as in
R, being a success (or the one given by `level=`):

```{code-cell} ipython3
spec = MS(['Price', 'Income'], response=Binomial('US'), intercept=True).fit(Carseats)
y = spec.build_response(Carseats)
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

counts = MS(['x'], response=Binomial('cases', trials='size')).fit(D).build_response(D)
prop = MS(['x'], response=Binomial('cases', trials='size', output='proportion')).fit(D).build_response(D)
X = MS(['x'], intercept=True).fit_transform(D)
family = sm.families.Binomial()
(sm.GLM(counts, X, family=family).fit().params,
 sm.GLM(prop['cases/size'], X, family=family, var_weights=prop['size']).fit().params)
```

## Multinomial responses

`Multinomial('ShelveLoc')` gives the class of each row as a categorical
whose categories are the levels seen at fit, and with `output='onehot'`
an indicator column for each level; `Multinomial([...])` takes a column
of counts for each class.

```{code-cell} ipython3
spec = MS(['Price', 'Sales'], response=Multinomial('ShelveLoc', output='onehot')).fit(Carseats)
spec.build_response(Carseats).iloc[:5]
```
