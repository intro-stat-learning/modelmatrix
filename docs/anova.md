---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# ANOVA using `ModelSpec`


We illustrate how to run specific ANOVA analyses
using `ModelSpec`, with statsmodels' `anova_lm`.

```{code-cell} ipython3
import numpy as np
import pandas as pd

from statsmodels.api import OLS
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from modelmatrix import ModelSpec, derived_feature
```

We will carry out two simple ANOVA analyses of the `Hitters` data.
We wish to predict a baseball player’s `Salary` on the
basis of various statistics associated with performance in the
previous year.

```{code-cell} ipython3
Hitters = load_data('Hitters')
np.isnan(Hitters['Salary']).sum()
```

    
 We see that `Salary` is missing for 59 players. The
`dropna()`  method of data frames removes all of the rows that have missing
values in any variable (by default --- see  `Hitters.dropna?`).

```{code-cell} ipython3
Hitters = Hitters.dropna()
Hitters.columns
```

## Grouping variables

A look at the [description](https://islp.readthedocs.io/en/latest/datasets/Hitters.html) of the data shows
that there are both career and 1986 offensive stats, as well as some defensive stats.

Let's group the offensive into recent and career offensive stats, as well as a group of defensive variables.

```{code-cell} ipython3
confounders = derived_feature(['Division', 'League', 'NewLeague'],
                              name='confounders')
offense_career = derived_feature(['CAtBat', 'CHits', 'CHmRun', 'CRuns', 'CRBI', 'CWalks'],
                                 name='offense_career')
offense_1986 = derived_feature(['AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Walks'],
                               name='offense_1986')
defense_1986 = derived_feature(['PutOuts', 'Assists', 'Errors'],
                               name='defense_1986')
```

We'll first do a sequential ANOVA where terms are added sequentially

```{code-cell} ipython3
design = ModelSpec([confounders, offense_career, defense_1986, offense_1986], intercept=True).fit(Hitters)
Y = np.array(Hitters['Salary'])
X = design.transform(Hitters)
```

```{code-cell} ipython3
M = OLS(Y, X).fit()
M.summary().tables[1]
```

We'll first produce the sequential, or Type I ANOVA results. This builds up a model sequentially and compares
two successive models.

```{code-cell} ipython3
df = anova_lm(*[OLS(Y, D).fit() for D in design.build_sequence(Hitters, anova_type='sequential')])
df.index = design.names
df
```

We can similarly compute the Type II ANOVA results which drops each term and compares to the full model.

```{code-cell} ipython3
D_full = design.transform(Hitters)
OLS_full = OLS(Y, D_full).fit()
dfs = []
for d in design.build_sequence(Hitters, anova_type='drop'):
    dfs.append(anova_lm(OLS(Y,d).fit(), OLS_full).iloc[1:])
df = pd.concat(dfs)
df.index = design.names
df
```
