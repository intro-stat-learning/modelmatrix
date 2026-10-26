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
                         build_columns)
from .columns import Column

try:
    from ._version import version as __version__
except ImportError:  # not installed
    __version__ = 'unknown'

__all__ = ['ModelSpec', 'Feature', 'Column', 'Contrast', 'Ordinal',
           'poly', 'ns', 'bs', 'derived_feature', 'pca', 'contrast',
           'ordinal', 'build_columns']
