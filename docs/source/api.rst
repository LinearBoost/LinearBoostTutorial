API Reference
=============

This page documents the public API of the **linearboost** package. Install it with
``pip install linearboost`` so that the classes below are available.

LinearBoostClassifier
---------------------

.. autoclass:: linearboost.LinearBoostClassifier
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

   The main ensemble classifier. Supports AdaBoost and gradient boosting,
   kernels (linear, RBF, poly, sigmoid), kernel approximation (RFF, Nyström),
   early stopping, subsampling, shrinkage, and class weighting.

SEFR
----

.. autoclass:: linearboost.SEFR
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

   The base binary linear classifier used by LinearBoost. Can be used standalone
   with linear or kernel (RBF, poly, sigmoid, precomputed) options. Supports
   ``fit_intercept`` and is fully scikit-learn compatible.
