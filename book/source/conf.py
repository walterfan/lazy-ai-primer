# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'AI 时代的软件工程：从敏捷开发到氛围编程'
copyright = '2026, Walter Fan'
author = 'Walter Fan'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
]

# Mermaid configuration
mermaid_version = "11"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme:'default'});"

# MyST Parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = 'AI 时代的软件工程'

html_theme_options = {
    "repository_url": "https://github.com/walterfan/ai-software-engineering-book",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

# -- Options for LaTeX/PDF output --------------------------------------------
latex_elements = {
    'preamble': r'''
\usepackage{xeCJK}
''',
}
