import os

from setuptools import find_packages, setup

# determining the directory containing setup.py
setup_path = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(setup_path, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

with open(os.path.join(setup_path, "requirements.txt")) as f:
    package_used = f.read().splitlines()

setup(
    # package information
    name='Captchami',
    packages=find_packages(),
    version='0.0.0',
    description='A little AI to solve simple captcha',
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=["Programming Language :: Python :: 3"],
    license='MIT',
    url='https://github.com/netphantom/captchami',
    keywords='',

    # author information
    author='netphantom',

    # installation info and requirements
    install_requires=package_used,
    setup_requires=package_used,

    # package deployment info
    include_package_data=True,
    zip_safe=False,
)
