from setuptools import setup


def _read(fname):
    with open(fname) as fobj:
        return fobj.read()


setup(
    name='flowly',
    version='0.1.0',
    description='Simple transformation chains in python.',
    long_description=_read("Readme.md"),
    author='Christopher Prohm',
    author_email='mail@cprohm.de',
    license='MIT',
    packages=["flowly"],
    tests_require=['pytest', 'dask', 'pytest-pep8'],
    classifiers=[
        'Development Status :: 4 - Beta',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
)
