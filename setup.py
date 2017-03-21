from setuptools import find_packages, setup
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

import importlib
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

console_scripts = [
    'train-as-auto-sklearn=as_auto_sklearn.train_as_auto_sklearn:main',
    'test-as-auto-sklearn=as_auto_sklearn.test_as_auto_sklearn:main',
    'validate-as-auto-sklearn=as_auto_sklearn.validate_as_auto_sklearn:main'
]

external_requirements = [
    'cython',
    'numpy',
    'scipy',
    'pandas',
    'joblib',
    'matplotlib',
    'sklearn',
    'docopt',
    'tqdm',
    'seaborn',
    'misc', # handled by requirements.txt now
    'auto-sklearn', # handled by requirements.txt
    'autofolio'
]

def _post_install(self):
    import site
    importlib.reload(site)
    
    import misc.utils as utils

    # we could check for existing programs, etc.
    pass


def install_requirements(is_user):
    # everything is now handled by requirements.txt
    pass

class my_install(_install):
    def run(self):
        level = logging.getLevelName("INFO")
        logging.basicConfig(level=level,
            format='%(levelname)-8s : %(message)s')

        _install.run(self)
        install_requirements(self.user)
        _post_install(self)

class my_develop(_develop):  
    def run(self):
        level = logging.getLevelName("INFO")
        logging.basicConfig(level=level,
            format='%(levelname)-8s : %(message)s')

        _develop.run(self)
        install_requirements(self.user)
        _post_install(self)

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='as_auto_sklearn',
        version='1.0.1',
        description="This project incorporates the auto-sklearn toolkit into "
            "a solver runtime prediction framework. The predictions directly "
            "yield a solution to the algorithm selection problem.",
        long_description=readme(),
        keywords="runtime prediction algorithm selection",
        url="",
        author="Brandon Malone",
        author_email="bmmalone@gmail.com",
        license='MIT',
        packages=find_packages(),
        install_requires = external_requirements,
        cmdclass={'install': my_install,  # override install
                  'develop': my_develop   # develop is used for pip install -e .
        },
        include_package_data=True,
        test_suite='nose.collector',
        tests_require=['nose'],
        entry_points = {
            'console_scripts': console_scripts
        },
        zip_safe=False
        )
