"""
Model matrices from tabular data, as scikit-learn transformers.
"""

from .model_spec import (ModelSpec,
                         Feature,
                         Contrast,
                         Ordinal,
                         poly,
                         ns,
                         bs,
                         derived_feature,
                         pca,
                         contrast,
                         ordinal,
                         build_columns,
                         wildcard,
                         Wildcard)
from .columns import Column
from .response import Response, Surv, Binomial, Multinomial
from .outcome import Outcome

try:
    from ._version import version as __version__
except ImportError:  # not installed
    __version__ = 'unknown'

__all__ = ['ModelSpec', 'Feature', 'Column', 'Contrast', 'Ordinal',
           'poly', 'ns', 'bs', 'derived_feature', 'pca', 'contrast',
           'ordinal', 'build_columns', 'wildcard', 'Wildcard',
           'Response', 'Surv', 'Binomial', 'Multinomial', 'Outcome']
