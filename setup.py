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
    'train-oasc-models=as_asl.train_oasc_models:main',
    'evaluate-oasc-models=as_asl.evaluate_oasc_models:main',
    'process-oasc-scenario=as_asl.process_oasc_scenario:main',
    'distribute-process-oasc-scenario=as_asl.distribute_process_oasc_scenario:main',
    'create-evaluation-summary=as_asl.create_evaluation_summary:main'
]

external_requirements = [
    'cython',
    'numpy',
    'scipy',
    'pandas',
    'joblib',
    'matplotlib',
    'sklearn',
    'mlxtend',
    'docopt',
    'tqdm',
    'seaborn',
    'misc', # handled by requirements.txt 
    'auto-sklearn', # handled by requirements.txt
    'aslib_scenario' # handled by requirements.txt
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

def description():
    description = ("This project incorporates the auto-sklearn toolkit (`asl`) "
        "into a solver runtime prediction framework. The predictions are then "
        "passed to a second classification model which yields a solution to "
        "the algorithm selection problem (`as`).")
    return description

setup(name='as_asl',
        version='1.1.0',
        description=description(),
        long_description=readme(),
        keywords="runtime prediction algorithm selection",
        url="https://github.com/bmmalone/as-asl",
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
