from distutils.core import setup
setup(
    name="Plot Options",
    version="0.0.1",
    packages=['dash=1', 'plotly',
	     'dash-bootstrap-components',
	     'dash-core-components',
	     'numpy',
	     'sympy',
	     'pandas']
    license="Private"
    long_description=open('code.org').read()
