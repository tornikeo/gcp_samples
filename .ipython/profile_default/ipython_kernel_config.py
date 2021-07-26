"""IPython configuration file."""

# pylint: disable=undefined-variable
c.IPKernelApp.extensions = [
    'caip_notebooks_serverextension',
    'google.cloud.bigquery',
    'sql'
]
c.InteractiveShellApp.matplotlib = 'inline'
