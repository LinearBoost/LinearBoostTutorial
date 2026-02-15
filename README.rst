LinearBoost Tutorial & Documentation
===================================

This repository contains the **tutorial and Sphinx documentation** for the
`LinearBoost <https://github.com/LinearBoost/linearboost-classifier>`_ classifier.

- **LinearBoost** is a fast, accurate binary classifier that boosts the linear
  base estimator **SEFR** (Scalable, Efficient, Fast Classifier).
- Documentation is built with Sphinx and can be hosted on Read the Docs.

Quick links
-----------

- **Install the classifier**: ``pip install linearboost``
- **Documentation**: https://linearboost.readthedocs.io/
- **Source & issues**: https://github.com/LinearBoost/linearboost-classifier

Building the docs locally
-------------------------

From this directory (LinearBoostTutorial)::

   pip install -r docs/requirements.txt
   cd docs && make html

Open ``docs/_build/html/index.html`` in a browser.

Read the Docs
-------------

The project is configured for Read the Docs via ``.readthedocs.yaml``. The build
installs dependencies from ``docs/requirements.txt`` (which includes ``linearboost``)
and builds Sphinx from ``docs/source/conf.py``.

License
-------

Same as the main LinearBoost project (see the main repository).
