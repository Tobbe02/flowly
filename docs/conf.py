extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.4', None),
    'tooz': ('https://toolz.readthedocs.io/en/latest/', None),
    'dask': ('http://dask.pydata.org/en/latest/', None),
}

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'Flowly'
copyright = u'2016, Christopher Prohm'
author = u'Christopher Prohm'

version = '0.1'
release = '0.1'

language = None
exclude_patterns = ['_build']
pygments_style = 'sphinx'
todo_include_todos = False

html_theme = 'alabaster'
html_static_path = ['_static']
htmlhelp_basename = 'Flowlydoc'
