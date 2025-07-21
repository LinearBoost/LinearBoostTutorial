Usage Guide
===========

This guide provides a comprehensive overview of the `LinearBoostClassifier`, from basic setup to advanced features.

.. _installation:

Installation
------------

First, install the LinearBoost library using pip. It's recommended to do this within a virtual environment.

.. code-block:: console

   (.venv) $ pip install linearboost

---

A Complete Example
------------------

Let's walk through a full example, from generating synthetic data to training the model and evaluating its performance. This demonstrates the standard workflow.

.. code-block:: python

   from linearboost import LinearBoostClassifier
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   # 1. Generate a synthetic dataset
   # X will have 20 features, y will be binary {0, 1}
   X, y = make_classification(
       n_samples=1000,
       n_features=20,
       n_informative=10,
       n_redundant=5,
       random_state=42
   )

   # 2. Split data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # 3. Initialize and train the LinearBoostClassifier
   # We'll use the default parameters for this first example
   clf = LinearBoostClassifier()
   clf.fit(X_train, y_train)

   # 4. Make predictions on the test set
   y_pred = clf.predict(X_test)

   # 5. Evaluate the model's accuracy
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Model Accuracy: {accuracy:.4f}")
   # Expected output: Model Accuracy: 0.8800 or similar

   # You can also predict probabilities
   y_proba = clf.predict_proba(X_test)
   print("Probability estimates for the first 2 samples:")
   print(y_proba[:2])

---

Key Parameters in Detail
------------------------

The power of `LinearBoostClassifier` lies in its customizable parameters. Understanding them is key to tuning the model for your specific dataset.

### Boosting Parameters

These parameters control the core AdaBoost algorithm.

-   **n_estimators** : `int`, default=`200`
    This is the maximum number of `SEFR` estimators to train in sequence. Boosting will stop early if a perfect fit is achieved. A larger number of estimators can lead to a more complex model but also risks overfitting.

-   **learning_rate** : `float`, default=`1.0`
    This parameter shrinks the contribution of each classifier. There is a **trade-off** between `learning_rate` and `n_estimators`: a lower learning rate requires more estimators to achieve the same level of performance but can lead to better generalization. Values are typically between `0.0` and `1.0`.

-   **algorithm** : `{'SAMME', 'SAMME.R'}`, default=`'SAMME.R'`
    This specifies the AdaBoost variant to use.
    -   `'SAMME.R'` requires the base estimator to have a `predict_proba` method. It typically converges faster and achieves a lower test error with fewer boosting iterations. Since the base `SEFR` estimator supports probability prediction, **'SAMME.R' is the recommended and default choice**.
    -   `'SAMME'` is a discrete version that can be used with classifiers that don't predict probabilities.

### Data Scaling

`LinearBoostClassifier` automatically scales your data. You just need to choose the method.

-   **scaler** : `str`, default=`'minmax'`
    Specifies which scikit-learn scaler to apply. The data is first transformed by the chosen scaler and then by a `MinMaxScaler` to ensure all values are in the `[0, 1]` range required by the base `SEFR` estimator.
    -   `'minmax'`: Best for standard data without significant outliers.
    -   `'standard'`: Centers data to have a mean of 0 and a standard deviation of 1. Good for data that follows a Gaussian distribution.
    -   `'robust'`: Uses medians and quartiles, making it robust to outliers. Use this if your dataset contains significant anomalies.
    -   `'quantile-uniform'` or `'quantile-normal'`: Non-linear transformations that can help spread out concentrated values and handle non-Gaussian distributions.
    -   `'power'`: Applies a power transformation to make data more Gaussian-like.
    -   Others include `'normalizer-l1'`, `'normalizer-l2'`, `'maxabs'`.

### Handling Imbalanced Data

-   **class_weight** : `dict`, `list of dicts`, or `'balanced'`, default=`None`
    Use this parameter to give more importance to under-represented classes.
    -   `'balanced'`: Automatically adjusts weights to be inversely proportional to class frequencies. For a dataset with 75 samples of class 0 and 25 of class 1, it would implicitly use weights like `{0: 0.67, 1: 2.0}`.
    -   `{0: 1, 1: 10}`: Manually sets the weight for class 1 to be 10 times that of class 0.

### Non-linear Classification with Kernels

These parameters are passed to the underlying `SEFR` estimator to enable it to learn non-linear decision boundaries using the "kernel trick".

-   **kernel** : `{'linear', 'poly', 'rbf', 'sigmoid'}`, default=`'linear'`
    -   `'linear'`: Creates a standard linear separator. Fast and effective for linearly separable data.
    -   `'rbf'` (Radial Basis Function): A powerful and popular choice for capturing complex, non-linear patterns. Its flexibility is controlled by `gamma`.
    -   `'poly'`: Can find polynomial decision boundaries. Its complexity is controlled by `degree`.
-   **gamma** : `float`, default=`None` (interpreted as `1 / n_features`)
    Kernel coefficient for `'rbf'` and `'poly'`. It defines how much influence a single training example has. A low `gamma` value creates a smooth, broad decision boundary, while a high `gamma` value creates a more complex, tight-fitting boundary that can lead to overfitting.
-   **degree** : `int`, default=`3`
    The degree of the polynomial for the `'poly'` kernel.
-   **coef0** : `float`, default=`1`
    An independent term in the `'poly'` and `'sigmoid'` kernels.

.. code-block:: python

   # Example using a non-linear RBF kernel for complex data
   model = LinearBoostClassifier(
       kernel='rbf',
       gamma=0.1,         # Custom gamma value
       n_estimators=100,
       learning_rate=0.5
   )
   model.fit(X_train, y_train)

---

Advanced Usage
--------------

### Inspecting the Fitted Model

After fitting, you can inspect the model's attributes to understand its components.

.. code-block:: python

   # Assuming 'clf' is a fitted LinearBoostClassifier
   print(f"Number of estimators trained: {len(clf.estimators_)}")

   # Weights of each estimator in the ensemble
   print(f"Estimator weights: {clf.estimator_weights_}")

   # Classification error for each estimator
   print(f"Estimator errors: {clf.estimator_errors_}")

   # The fitted scaler object can be inspected or used
   print(f"Fitted scaler: {clf.scaler_}")

   # You can use the fitted scaler to transform new data independently
   X_new_transformed = clf.scaler_.transform(X_test)

### Hyperparameter Tuning with GridSearchCV

`LinearBoostClassifier` is compatible with the scikit-learn ecosystem, so you can use tools like `GridSearchCV` to find the best parameters.

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   # Define the parameter grid to search
   param_grid = {
       'n_estimators': [50, 100, 200],
       'learning_rate': [0.1, 0.5, 1.0],
       'kernel': ['linear', 'rbf'],
       'scaler': ['standard', 'robust']
   }

   # Initialize the classifier and the grid search
   lbc = LinearBoostClassifier()
   grid_search = GridSearchCV(estimator=lbc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

   # Fit the grid search to the data
   grid_search.fit(X_train, y_train)

   # Print the best parameters found
   print(f"Best parameters found: {grid_search.best_params_}")

   # The best estimator is already fitted and can be used directly
   best_model = grid_search.best_estimator_
   accuracy = best_model.score(X_test, y_test)
   print(f"Tuned Model Accuracy: {accuracy:.4f}")

---

Limitations
-----------

-   **Binary Classification Only**: The current version is designed exclusively for two-class problems.
-   **Numeric Features Only**: The input features (`X`) must be numeric. Categorical features need to be encoded (e.g., via one-hot encoding) before being passed to the model.

---

Feedback
--------

For more details, please refer to the [GitHub Repo](https://github.com/LinearBoost/linearboost-classifier). We welcome contributions, issues, and suggestions!
