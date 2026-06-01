# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Include the project root directory

# Add the nnpiv directory and its subdirectories to sys.path
sys.path.insert(0, os.path.abspath('../nnpiv'))  
nnpiv_dir = os.path.abspath('../nnpiv')
for root, dirs, files in os.walk(nnpiv_dir):
    sys.path.insert(0, root)

# -- Project information -----------------------------------------------------

project = "Nested Nonparametric Instrumental Variable Regression"
copyright = '2025, Meza, I. and Singh, R.'
author = "Isaac Meza and Rahul Singh"
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# -- General configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx_rtd_dark_mode",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


autodoc_mock_imports = ["scipy", "sklearn", "statsmodels", "tqdm", "copy",
                        "torch", "mliv", "mliv.linear", "joblib", "matplotlib"]

# Pseudo-type strings used in NumPy-style docstrings that are not Python objects.
nitpick_ignore = [
    ("py:class", "'auto'"),
    ("py:class", "'identity'}"),
    ("py:class", "'polynomial'}"),
    ("py:class", "1"),
    ("py:class", "2D array-like"),
    ("py:class", "DataFrame"),
    ("py:class", "Same as"),
    ("py:class", "array-like"),
    ("py:class", "bool /"),
    ("py:class", "boolean"),
    ("py:class", "callable"),
    ("py:class", "d_a"),
    ("py:class", "d_c"),
    ("py:class", "d_cp"),
    ("py:class", "default 'rff'"),
    ("py:class", "default=False"),
    ("py:class", "default='auto'"),
    ("py:class", "default='rff'"),
    ("py:class", "default='sigma_i'"),
    ("py:class", "default=1.0"),
    ("py:class", "default=1e-10"),
    ("py:class", "default=1e-6"),
    ("py:class", "default=1e-8"),
    ("py:class", "default=123"),
    ("py:class", "default=3"),
    ("py:class", "default=300"),
    ("py:class", "default=5.0"),
    ("py:class", "default=True"),
    ("py:class", "estimator"),
    ("py:class", "estimator /"),
    ("py:class", "iterable"),
    ("py:class", "mapping"),
    ("py:class", "n"),
    ("py:class", "optional"),
    ("py:class", "p"),
    ("py:class", "shape"),
    ("py:class", "{'rff'"),
    ("py:class", "{'sigma_i'"),
]
