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
First, import linearboost in your project.

   >>> from linearboost import LinearBoostClassifier

Then, you can import an instance, and train it on your data.
   
   >>> model = LinearBoostClassifier()
   >>> model.fit(X_train, y_train)
   >>> y_pred = model.predict(X_test)

You can also have predict probabilities:

   >>> y_pred_proba = model.predict_proba(X_test)

At the moment, the algorithm only supports binary classification. It also accepts only numeric features. Support for categorical features and multi-label classification are in future plans.

Parameters
----------

You can initialize the model with parameters. The main parameter is the number of estimators. As the model converges faster than similar methods, you generally need less estimators.

   >>> model = LinearBoostClassifier(n_estimators=5)

You can also set learning rate:

   >>> model = LinearBoostClassifier(learning_rate=0.5)

The random state can also be set:

   >>> model = LinearBoostClassifier(random_state=42)

As the SEFR classifier should only get non-negative values, we should scale the data before handing them to SEFR. Here, you have a set of scalers to choose from. They 'quantile-uniform' and 'quantile-normal' (corresponding to QuantileTransformer), 'normalizer-l1', 'normalizer-l2', and 'normalizer-max' (corresponding to Normalizer and the norm), 'standard' (corresponding to Standard Scaler), 'power' (corresponding to PowerTransformer with yeo-johnson method), 'maxabs' (corresponding to MaxAbsScaler), and 'robust' (corresponding to RobustScaler).

   >>> model = LinearBoostClassifier(scaler="quantile-uniform")

And last, but not least, you can choose between two algorithms for boosting, SAMME and SAMME.R (see the `documentation for scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html>`_).

   >>> model = LinearBoostClassifier(algorithm='SAMME')
   >>> model = LinearBoostClassifier(algorithm='SAMME.R')

