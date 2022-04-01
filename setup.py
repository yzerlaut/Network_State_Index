from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='NSI',
    version='1.0',
    description='Classifying Network State from LFP in Neocortex',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yzerlaut/Network_State_Index',
    author='Yann Zerlaut',
    author_email='yann.zerlaut@icm-institute.org',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    keywords='LFP, cortex, network state',
    packages=find_packages(),
    install_requires=[
        "numpy",
        # "pynwb",
        # "pyabf",
        # "argparse",
        # "pyqt5",
        # "pyqtgraph",
        # "neo",
        "scipy"
    ]
)
