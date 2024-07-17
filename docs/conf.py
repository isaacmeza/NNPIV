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
copyright = '2024, Meza, I. and Singh, R.'
author = "Isaac Meza and Rahul Singh"
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# -- General configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx_rtd_dark_mode",
    "nbsphinx",
    "sphinx.ext.duration",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

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


autodoc_mock_imports = ["numpy", "scipy", "sklearn", "statsmodels", "tqdm", "copy", 
                        "torch", "mliv", "mliv.linear", "joblib"]
