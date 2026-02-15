Usage Guide
===========

This guide covers **LinearBoostClassifier** and the base estimator **SEFR** from installation through advanced options, aligned with the current implementation in ``linear_boost.py`` and ``sefr.py``.

.. _installation:

Installation
------------

Install the **linearboost** package (and optionally use a virtual environment):

.. code-block:: console

   pip install linearboost

Requirements: Python >= 3.8, scikit-learn >= 1.2.2.

---

A Complete Example
------------------

Basic workflow: load or generate data, split, fit **LinearBoostClassifier**, and evaluate.

.. code-block:: python

   from linearboost import LinearBoostClassifier
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, f1_score

   X, y = make_classification(
       n_samples=1000,
       n_features=20,
       n_informative=10,
       n_redundant=5,
       random_state=42,
   )

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   clf = LinearBoostClassifier()
   clf.fit(X_train, y_train)

   y_pred = clf.predict(X_test)
   print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
   print(f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

   y_proba = clf.predict_proba(X_test)
   print("Probability estimates for first 2 samples:")
   print(y_proba[:2])

---

Key Parameters (LinearBoostClassifier)
--------------------------------------

Parameters below match the current API. See :doc:`api` for the full signature and attributes.

Boosting type and algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **boosting_type** : `{'adaboost', 'gradient'}`, default=`'adaboost'`
  - ``'adaboost'``: Classic AdaBoost (SAMME or SAMME.R) that reweights samples by classification error.
  - ``'gradient'``: Gradient boosting; each base estimator fits pseudo-residuals (negative gradient of log-loss). Often better for highly non-linear or XOR-like patterns. When using ``'gradient'``, the **algorithm** parameter is ignored.

- **algorithm** : `{'SAMME', 'SAMME.R'}`, default=`'SAMME.R'`
  Used only when **boosting_type='adaboost'**. ``'SAMME.R'`` typically converges faster and achieves lower test error with fewer iterations. ``'SAMME'`` is the discrete variant.

- **n_estimators** : `int`, default=`200`
  Maximum number of base SEFR estimators. With **early_stopping=True** you can set a larger value (e.g. 500) and let training stop when validation score does not improve.

- **learning_rate** : `float`, default=`1.0`
  Shrinks the contribution of each estimator. There is a trade-off with **n_estimators**: lower learning rate usually needs more estimators but can improve generalization.

Regularization and early stopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **subsample** : `float`, default=`1.0`
  Fraction of samples used to fit each base estimator. Values in (0, 1] enable stochastic boosting (e.g. ``0.8``) and can reduce variance.

- **shrinkage** : `float`, default=`1.0`
  Multiplier for each estimator’s weight. Values in (0, 1] (e.g. ``0.8--0.95``) reduce overfitting and can improve generalization.

- **early_stopping** : `bool`, default=`False`
  If ``True``, training stops when validation score does not improve for **n_iter_no_change** consecutive iterations. Requires **n_iter_no_change** to be set.

- **validation_fraction** : `float`, default=`0.1`
  Fraction of training data used as validation for early stopping. Only used when **early_stopping=True** and **subsample >= 1.0**. When **subsample < 1.0**, out-of-bag (OOB) evaluation is used instead and this parameter is ignored.

- **n_iter_no_change** : `int`, default=`5`
  Number of iterations with no improvement to wait before stopping (when **early_stopping=True**).

- **tol** : `float`, default=`1e-4`
  Minimum improvement in score to count as “improvement” for early stopping.

Data scaling
~~~~~~~~~~~~

- **scaler** : `str`, default=`'minmax'`
  Scaling applied before training. When ``scaler != 'minmax'``, the pipeline is: chosen scaler → **MinMaxScaler** (so SEFR always sees values in a bounded range). Options include:
  - ``'minmax'``: MinMaxScaler only.
  - ``'standard'``, ``'robust'``, ``'quantile-uniform'``, ``'quantile-normal'``.
  - ``'normalizer-l1'``, ``'normalizer-l2'``, ``'normalizer-max'``, ``'power'``, ``'maxabs'``.

The fitted transformer is available as **scaler_** (a pipeline when scaler is not ``'minmax'``).

Imbalanced data and custom loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **class_weight** : `dict`, `'balanced'`, or `None`, default=`None`
  Class weights. Use ``'balanced'`` to weight inversely to class frequencies. Can be combined with **sample_weight** in ``fit()``.

- **loss_function** : callable or `None`, default=`None`
  Optional custom loss with signature ``(y_true, y_pred, sample_weight) -> float`` for optimization.

Kernels and kernel approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **kernel** : `{'linear', 'poly', 'rbf', 'sigmoid'}` or callable, default=`'linear'`
  - ``'linear'``: No kernel; fastest, for linearly separable data.
  - ``'rbf'``: Radial basis function; flexible for non-linear boundaries.
  - ``'poly'``: Polynomial; complexity controlled by **degree**.
  - ``'sigmoid'``: Sigmoid kernel.

- **gamma** : `float` or `None`, default=`None`
  Kernel coefficient for ``'rbf'``, ``'poly'``, ``'sigmoid'``. If ``None``, set to ``1 / n_features``.

- **degree** : `int`, default=`3`
  Degree for ``'poly'`` kernel.

- **coef0** : `float`, default=`1`
  Independent term in ``'poly'`` and ``'sigmoid'`` kernels.

- **kernel_approx** : `{'rff', 'nystrom'}` or `None`, default=`None`
  For large datasets with non-linear **kernel**, use approximation to avoid an O(n²) Gram matrix:
  - ``'rff'``: Random Fourier Features; only valid for **kernel='rbf'**.
  - ``'nystrom'``: Nyström approximation; works with ``'rbf'``, ``'poly'``, ``'sigmoid'``.
  - ``None``: Exact kernel (full Gram matrix).

- **n_components** : `int`, default=`256`
  Dimensionality of the kernel feature map when **kernel_approx** is used (number of random features for ``'rff'``, rank for ``'nystrom'``).

---

Examples by feature
--------------------

Non-linear kernel (exact)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = LinearBoostClassifier(
       kernel="rbf",
       gamma=0.1,
       n_estimators=100,
       learning_rate=0.5,
   )
   model.fit(X_train, y_train)

Kernel approximation (scalable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = LinearBoostClassifier(
       kernel="rbf",
       kernel_approx="rff",
       n_components=256,
       n_estimators=100,
   )
   model.fit(X_train, y_train)

Gradient boosting
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = LinearBoostClassifier(
       boosting_type="gradient",
       kernel="rbf",
       n_estimators=200,
   )
   model.fit(X_train, y_train)

Early stopping with validation split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = LinearBoostClassifier(
       n_estimators=500,
       early_stopping=True,
       validation_fraction=0.1,
       n_iter_no_change=5,
       tol=1e-4,
   )
   model.fit(X_train, y_train)

Early stopping with OOB (when using subsampling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = LinearBoostClassifier(
       n_estimators=500,
       subsample=0.8,
       early_stopping=True,
       n_iter_no_change=5,
   )
   model.fit(X_train, y_train)

Imbalanced data
~~~~~~~~~~~~~~~

.. code-block:: python

   model = LinearBoostClassifier(class_weight="balanced", n_estimators=200)
   model.fit(X_train, y_train)

---

Using SEFR standalone
----------------------

**SEFR** is the base binary linear classifier. You can use it alone for a very fast, lightweight model. It supports **fit_intercept**, **kernel** (linear, poly, rbf, sigmoid, precomputed), and **gamma**, **degree**, **coef0**.

.. code-block:: python

   from linearboost import SEFR

   clf = SEFR(kernel="rbf", fit_intercept=True)
   clf.fit(X_train, y_train)
   clf.predict(X_test)
   clf.score(X_test, y_test)

See :doc:`api` for SEFR’s full parameters and attributes.

---

Advanced usage
--------------

Inspecting the fitted model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(f"Number of estimators: {len(clf.estimators_)}")
   print(f"Estimator weights: {clf.estimator_weights_}")
   print(f"Estimator errors: {clf.estimator_errors_}")
   print(f"Fitted scaler: {clf.scaler_}")

   # Transform new data with the same scaling
   X_new_scaled = clf.scaler_.transform(X_test)

When **boosting_type='gradient'**, the raw scores and initial score are in **F_** and **init_score_** (if present).

Hyperparameter tuning with GridSearchCV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   param_grid = {
       "n_estimators": [50, 100, 200],
       "learning_rate": [0.1, 0.5, 1.0],
       "kernel": ["linear", "rbf"],
       "scaler": ["minmax", "robust"],
       "boosting_type": ["adaboost", "gradient"],
   }

   lbc = LinearBoostClassifier()
   grid = GridSearchCV(estimator=lbc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
   grid.fit(X_train, y_train)

   print(grid.best_params_)
   best_model = grid.best_estimator_
   print(f"Test accuracy: {best_model.score(X_test, y_test):.4f}")

---

Limitations
-----------

- **Binary classification only**: Both LinearBoostClassifier and SEFR support only two-class targets.
- **Numeric features only**: Input ``X`` must be numeric. Encode categorical features (e.g. one-hot) before use.

---

Feedback
--------

For more details and source code, see the `LinearBoost GitHub repository <https://github.com/LinearBoost/linearboost-classifier>`_. We welcome issues, contributions, and suggestions.
