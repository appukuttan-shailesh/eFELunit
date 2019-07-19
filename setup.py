try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='eFELunit',
    version='1.1.2',
    author='Shailesh Appukuttan, Andrew Davison',
    author_email='shailesh.appukuttan@unic.cnrs-gif.fr, andrew.davison@unic.cnrs-gif.fr',
    packages=['eFELunit',
              'eFELunit.capabilities',
              'eFELunit.tests',
              'eFELunit.scores',
              'eFELunit.plots'],
    url='https://github.com/appukuttan-shailesh/eFELunit',
    keywords = ['eFEL', 'electrophysiology', 'electrical', 'testing', 'validation framework'],
    license='MIT',
    description='A SciUnit library for data-driven testing of neuronal morphologies.',
    long_description="",
    install_requires=['neo','elephant','sciunit>=0.1.5.2',],
    dependency_links = ['git+http://github.com/neuralensemble/python-neo.git#egg=neo-0.4.0dev',
                        'https://github.com/scidash/sciunit/tarball/dev']
)
