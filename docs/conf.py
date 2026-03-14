project = "simulacra"
copyright = "2025, Elliot Hill"
author = "Elliot Hill"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "numpydoc",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- MyST -------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

# -- Autodoc / autosummary --------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

# -- numpydoc ----------------------------------------------------------------
numpydoc_show_class_members = False

# -- Plot directive ----------------------------------------------------------
plot_include_source = True
plot_html_show_source_link = False
plot_formats = [("png", 150)]
plot_rcparams = {
    "figure.figsize": (5.5, 3.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.alpha": 0.3,
    "font.size": 10,
}

# -- Doctest -----------------------------------------------------------------
doctest_global_setup = """
import torch
import simulacra
"""

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML theme --------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "navbar_align": "left",
    "github_url": "https://github.com/elliothill/simulacra",
}
