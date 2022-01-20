# Configuration file for the Sphinx documentation builder.

import sys
# import pathlib
# sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
import os
sys.path.insert(0, os.path.abspath('../../pymsm/'))
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information

project = 'PyMSM'
copyright = '2022, Rossman & Keshet'
author = 'Hagai Rossman, Ayya Keshet'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "nbsphinx",
]

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_flags = [
    # Make sure that any autodoc declarations show the right members
    "members",
    "inherited-members",
    "show-inheritance",
]
autosummary_generate = True  # Make _autosummary files and include them