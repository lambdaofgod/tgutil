from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='tgutil',
    version='0.1',
    description='Python helpers for text generation',
    url='https://github.com/lambdaofgod/tgutil',
    author='Jakub Bartczuk',
    packages=find_packages(),
    install_requires=requirements
)
