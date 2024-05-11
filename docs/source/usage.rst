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

Then, you can import an instance:
   
   >>> model = LinearBoostClassifier()
   >>> model.fit(X_train, y_train)
   >>> y_pred = model.predict(X_test)




To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

