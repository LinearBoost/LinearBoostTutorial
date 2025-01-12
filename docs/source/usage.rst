Usage
=====

.. _installation:

Installation
------------

To use LinearBoost, first install it using pip:

.. code-block:: console

   (.venv) $ pip install linearboost

Quick Start
-----------
Import the `LinearBoostClassifier` in your project:

.. code-block:: python

   >>> from linearboost import LinearBoostClassifier

Create an instance and train it on your dataset:

.. code-block:: python

   >>> model = LinearBoostClassifier()
   >>> model.fit(X_train, y_train)  # Train the model
   >>> y_pred = model.predict(X_test)  # Predict on test data

To get probability estimates instead of binary predictions:

.. code-block:: python

   >>> y_pred_proba = model.predict_proba(X_test)

**Note**: Ensure the features are numeric and properly scaled before training.

Key Parameters
--------------

When initializing the `LinearBoostClassifier`, you can customize the following parameters:

1. **n_estimators**:
   Number of boosting rounds (default: 200). Fewer estimators are typically required compared to other methods.
   Example:
   .. code-block:: python
      >>> model = LinearBoostClassifier(n_estimators=5)

2. **learning_rate**:
   Controls the contribution of each estimator (default: 1.0). Smaller values may require more estimators.
   Example:
   .. code-block:: python
      >>> model = LinearBoostClassifier(learning_rate=0.5)

3. **random_state**:
   Ensures reproducibility by setting the random seed.
   Example:
   .. code-block:: python
      >>> model = LinearBoostClassifier(random_state=42)

4. **scaler**:
   Scales the data before training. Available options:
   - `quantile-uniform` (QuantileTransformer with uniform distribution)
   - `quantile-normal` (QuantileTransformer with normal distribution)
   - `normalizer-l1`, `normalizer-l2`, `normalizer-max` (Normalizer with different norms)
   - `standard` (StandardScaler)
   - `power` (PowerTransformer)
   - `maxabs` (MaxAbsScaler)
   - `robust` (RobustScaler)

   Example:
   .. code-block:: python
      >>> model = LinearBoostClassifier(scaler="quantile-uniform")

5. **class_weight**:
   Handles class imbalance. Options:
   - `"balanced"`: Automatically adjusts weights inversely proportional to class frequencies.
   - Custom dictionary: Define weights manually, e.g., `{0: 2.0, 1: 1.0}`.
   Example:
   .. code-block:: python
      >>> model = LinearBoostClassifier(class_weight="balanced")

6. **loss_function**:
   Define a custom loss function for the boosting process.
   Example:
   .. code-block:: python
      >>> model = LinearBoostClassifier(loss_function=my_loss_function)

7. **algorithm**:
   Choose between two boosting algorithms:
   - `SAMME`: For discrete classification.
   - `SAMME.R`: For real-valued predictions.
   Example:
   .. code-block:: python
      >>> model = LinearBoostClassifier(algorithm='SAMME.R')

Limitations
-----------

- **Binary Classification Only**: The algorithm currently supports only binary classification tasks.
- **Numeric Features Only**: Ensure all features are numeric. Future updates may include support for categorical features.
- **Scaling Required**: Data must be scaled before being fed into the classifier. Choose a scaler suitable for your data distribution.

Examples
--------

1. **Basic Usage**:
   .. code-block:: python

      >>> model = LinearBoostClassifier(n_estimators=10, learning_rate=0.1, scaler="standard")
      >>> model.fit(X_train, y_train)
      >>> predictions = model.predict(X_test)

2. **Class Weights**:
   .. code-block:: python

      >>> model = LinearBoostClassifier(class_weight={0: 1.0, 1: 2.0})
      >>> model.fit(X_train, y_train)

3. **Custom Loss Function**:
   Define your loss function:
   .. code-block:: python

      def custom_loss(y_true, y_pred, weights):
          return np.mean(weights * (y_true - y_pred) ** 2)

   Use it with the model:
   .. code-block:: python

      >>> model = LinearBoostClassifier(loss_function=custom_loss)
      >>> model.fit(X_train, y_train)

4. **Probability Prediction**:
   .. code-block:: python

      >>> y_proba = model.predict_proba(X_test)

Feedback
--------

For detailed documentation, refer to the GitHub Repo (https://github.com/LinearBoost/linearboost-classifier).
Contributions, issues, and suggestions are welcome!

---

This `.srt` file is designed to provide both an overview and detailed guidance on using `LinearBoostClassifier`, making it beginner-friendly yet comprehensive for advanced users.
