import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'bayesian-models'
copyright = '2023, Alexander Rodis'
author = 'Alexander Rodis'
release = '0.1.0'

extensions = [
    'sphinxcontrib.katex',
    'sphinx.ext.autodoc',
    ]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['static']
