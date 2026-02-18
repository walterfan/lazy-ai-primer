# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Project Name'  # TODO: Replace with actual project name
copyright = '2026, Your Organization'  # TODO: Replace with actual copyright
author = 'Your Team'  # TODO: Replace with actual author
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',           # MyST markdown parser
    'sphinx.ext.autodoc',    # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',   # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',   # Add links to source code
    'sphinx.ext.intersphinx',# Link to other project's documentation
    'sphinx.ext.todo',       # Support for todo items
    'sphinx.ext.graphviz',   # Support for Graphviz graphs
    'sphinx_copybutton',     # Add copy button to code blocks
    'sphinx_design',         # Design components (cards, tabs, etc.)
    'sphinxcontrib.mermaid', # Mermaid diagrams (priority)
    'sphinxcontrib.plantuml',# PlantUML diagrams (fallback)
]

# MyST parser configuration
myst_enable_extensions = [
    "amsmath",          # LaTeX math
    "colon_fence",      # ::: fence for directives
    "deflist",          # Definition lists
    "dollarmath",       # $...$ for inline math
    "fieldlist",        # Field lists
    "html_admonition",  # HTML admonitions
    "html_image",       # HTML images
    "linkify",          # Auto-detect URLs
    "replacements",     # Text replacements
    "smartquotes",      # Smart quotes
    "strikethrough",    # ~~strikethrough~~
    "substitution",     # {{ variable }}
    "tasklist",         # - [ ] task lists
]

# MyST parser settings
myst_heading_anchors = 3  # Auto-generate anchors for headings up to level 3
myst_footnote_transition = True
myst_dmath_double_inline = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Read the Docs theme
# html_theme = 'furo'  # Alternative: Modern, clean theme

html_static_path = ['_static']

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# Custom CSS
html_css_files = [
    'custom.css',
]

# Logo and favicon
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, 'project.tex', 'Project Documentation',
     'Your Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'project', 'Project Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'project', 'Project Documentation',
     author, 'project', 'Project description.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# sphinx.ext.intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# sphinx.ext.todo
todo_include_todos = True

# sphinx.ext.autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Custom configuration ----------------------------------------------------

# Add any paths that contain custom static files (such as style sheets)
# html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer.
html_show_sphinx = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# -- Mermaid configuration ---------------------------------------------------

# Mermaid version (use latest stable)
mermaid_version = "10.6.1"

# Mermaid initialization config
mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
    theme: 'default',
    securityLevel: 'loose',
    logLevel: 'error',
    flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
    },
    sequence: {
        useMaxWidth: true,
        diagramMarginX: 50,
        diagramMarginY: 10,
        actorMargin: 50,
        width: 150,
        height: 65,
        boxMargin: 10,
        boxTextMargin: 5,
        noteMargin: 10,
        messageMargin: 35,
        mirrorActors: true,
        bottomMarginAdj: 1,
        useMaxWidth: true,
        rightAngles: false,
        showSequenceNumbers: false
    },
    gantt: {
        titleTopMargin: 25,
        barHeight: 20,
        barGap: 4,
        topPadding: 50,
        leftPadding: 75,
        gridLineStartPadding: 35,
        fontSize: 11,
        numberSectionStyles: 4,
        axisFormat: '%Y-%m-%d'
    }
});
"""

# -- PlantUML configuration --------------------------------------------------

# PlantUML command (requires PlantUML jar or server)
# Option 1: Local jar file
# plantuml = 'java -jar /path/to/plantuml.jar'

# Option 2: PlantUML server (recommended for CI/CD)
plantuml = 'java -jar plantuml.jar'
plantuml_output_format = 'svg'

# Alternative: Use public PlantUML server (not recommended for production)
# plantuml_server_url = 'http://www.plantuml.com/plantuml'
# plantuml_output_format = 'svg'

# -- Graphviz configuration --------------------------------------------------

graphviz_output_format = 'svg'
