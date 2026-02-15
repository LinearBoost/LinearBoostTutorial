# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'LinearBoost'
copyright = '2025, Hamidreza Keshavarz & Reza Rawassizadeh'
author = 'Hamidreza Keshavarz'

release = '0.1.7'
version = '0.1.7'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# Ensure the linearboost package can be imported when building docs (e.g. when installed via docs/requirements.txt)
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'show-inheritance': True,
}
napoleon_use_param = True
napoleon_use_ivar = True

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
