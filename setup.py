from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(
    name='python-llm',
    version='0.1.7',
    author='Daniel Gross',
    author_email='d@dcgross.com',
    description='An LLM wrapper for Humans',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=['llm', 'llm.*']),
    url="https://github.com/danielgross/python-llm",
    classifiers=[
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
