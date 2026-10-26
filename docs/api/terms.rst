Terms
=====

Helpers that make terms
-----------------------

.. autofunction:: modelmatrix.poly
.. autofunction:: modelmatrix.ns
.. autofunction:: modelmatrix.bs
.. autofunction:: modelmatrix.pca
.. autofunction:: modelmatrix.contrast
.. autofunction:: modelmatrix.ordinal
.. autofunction:: modelmatrix.derived_feature
.. autofunction:: modelmatrix.build_columns

Terms and columns
-----------------

.. autoclass:: modelmatrix.Feature
.. autoclass:: modelmatrix.Column
    :members: get_columns

Encoders of categoricals
------------------------

.. autoclass:: modelmatrix.Contrast
    :members: fit, transform
.. autoclass:: modelmatrix.Ordinal
    :members: fit, transform
