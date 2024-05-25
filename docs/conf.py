# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../src/'))
sys.path.insert(1, os.path.abspath('../library/'))

project = 'Advanced Graph Clustering'
author = 'Timothe Watteau, Joaquim Jusseau, Aubin Bonnefoy, Simon Illouz-Laurent'
copyright = '2024, ' + author
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	"sphinx.ext.autodoc",
	"sphinx.ext.viewcode",
	"sphinx.ext.napoleon",
	"sphinx.ext.todo",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
