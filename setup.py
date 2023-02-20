"""Contains information about the package"""

import os
import codecs
import importlib.util as ut

from setuptools import setup, find_packages

setup_file_path = os.path.abspath(os.path.dirname(__file__))


def read_utf8(filename):
    """
    Ensure consistent encoding

    Args:
        filename (str): read me markup file name
    Returns:
        str
    """
    with codecs.open(os.path.join(setup_file_path, filename), encoding='utf-8') as f:
        return f.read()


def read_requirements(filename):
    """
    For a given requirements file, returns the lines of a file a list of strings

    Args:
        filename (str): requirements file name
    Returns:
        list
    """
    with open(filename) as file:
        return file.readlines()


def version_spec(filename):
    """
    Wrote method to import just the _version module, don't pull in extra dependencies

    Args:
        filename (str): _version.py file path

    Returns:
         func
    """
    return ut.spec_from_file_location('_version', filename)


version_file_path = os.path.join(setup_file_path, 'model', '_version.py')
spec = version_spec(version_file_path)
version_module = ut.module_from_spec(spec)
spec.loader.exec_module(version_module)
version = version_module.VERSION

install_requires = read_requirements('requirements.txt')
long_description = read_utf8('README.md')
short_description = "An approach to learn non linear relationship between option parameters and its price using /" \
                    "Deep Neural Network"""
url = 'https://github.com/Karanpalshekhawat/pricing-options-and-computing-implied-volatility-using-deep-neural-network'

setup(
    name='option-price-using-NN',
    version=version,
    description=short_description,
    long_description=long_description,
    url=url,
    author='Karan Pal Shekhawat',
    author_email='karanpal609@gmail.com',
    packages=find_packages(exclude=['tests', 'test.*', '*ipynb']),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
