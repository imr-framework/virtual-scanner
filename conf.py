# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Virtual Scanner'
copyright = '2019, Columbia University'
author = 'Gehua Tong, Sairam Geethanath, Enlin Qian, Keerthi Sravan Ravi, Marina Jimeno Manso'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon','sphinx.ext.autodoc']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Module naming
add_module_names = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "classic"
columbiablue = 'rgb(185,217,235)'
lightblue = 'rgb(108,172,228)'
brightblue = 'rgb(0,114,206)'
deepblue = 'rgb(0,51,160)'
darkblue = 'rgb(2,33,105)'
html_theme_options = {
    'sidebarbgcolor' : columbiablue,
    'footerbgcolor' : columbiablue,
    'relbarbgcolor' : columbiablue,
    'headbgcolor' : deepblue,
    'codebgcolor' : 'LemonChiffon',
    'codetextcolor' : 'black',

    'sidebartextcolor' : deepblue,
    'headtextcolor' : 'white',

    'linkcolor' : brightblue,
    'textcolor' : darkblue,
    'headlinkcolor': brightblue,
    'sidebarlinkcolor': brightblue,
    'relbarlinkcolor': brightblue,
    'visitedlinkcolor' : brightblue
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
