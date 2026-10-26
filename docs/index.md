# modelmatrix

`modelmatrix` builds model matrices from tabular data. Its `ModelSpec`
is a scikit-learn transformer: a list of terms (columns, categorical
encodings, polynomials, splines, principal components and their
interactions) is fit to a `pd.DataFrame`, a polars or pyarrow frame or
an `np.ndarray`, and transforms new data the same way. It goes in a
`Pipeline`, and the parameters of its terms are tuned with
`GridSearchCV`.

```
pip install modelmatrix
```

- [Building design matrices with `ModelSpec`](spec.md)
- Features: [polynomials](poly.md), [splines](splines.md) and [PCA](PCA.md)
- [Responses and the wildcard](responses.md): `y ~ .`, `Surv`, `Binomial` and `Multinomial`
- [ANOVA](anova.md) with `build_sequence`
- [`ModelSpec` compared](comparison.md) with `ColumnTransformer`, patsy and formulaic
- The [API reference](https://intro-stat-learning.github.io/modelmatrix/api/)

`ModelSpec` was written for the labs of
[An Introduction to Statistical Learning with Python](https://www.statlearning.com),
whose `ISLP` package provides the data used on these pages.
For more information, visit the [GitHub repository](https://github.com/intro-stat-learning/modelmatrix).
