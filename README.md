# modelmatrix

Model matrices from tabular data, as scikit-learn transformers.

`ModelSpec` builds a design matrix from a `pd.DataFrame` (or a polars or
pyarrow frame, or an `np.ndarray`) from a list of terms: columns,
categorical encodings, polynomials, splines, principal components and
their interactions. It is a scikit-learn transformer, so it goes in a
`Pipeline`, is cloned and cross-validated, and the parameters of its
terms are tuned with `GridSearchCV`.

```python
from modelmatrix import ModelSpec, ns, poly
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

spec = ModelSpec([ns('age', df=4), poly('year', 2), 'education',
                  ('age', 'education')])
X = spec.fit_transform(Wage)

pipe = make_pipeline(spec, LinearRegression())
grid = GridSearchCV(pipe, {'modelspec__terms__0__df': [2, 4, 8]})
grid.fit(Wage, Wage['wage'])
```

`ModelSpec` was written for the labs of
[An Introduction to Statistical Learning with Python](https://www.statlearning.com),
and `from ISLP.models import ModelSpec` still works there.

Documentation: https://intro-stat-learning.github.io/modelmatrix

## Install

```
pip install modelmatrix
```

Requires Python 3.11 or later. BSD licensed.
