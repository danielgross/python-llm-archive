from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='python-llm',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python package for llm',
    packages=find_namespace_packages(include=['llm', 'llm.*']),
    classifiers=[
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
