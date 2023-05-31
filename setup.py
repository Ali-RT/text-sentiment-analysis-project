#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Ali Rahim-Taleqani",
    author_email='ali.rahim.taleqani@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Train a simple neural network for analyzing the sentiment in the text of movie reviews.",
    entry_points={
        'console_scripts': [
            'text_sentiment_analysis=text_sentiment_analysis.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='text_sentiment_analysis',
    name='text_sentiment_analysis',
    packages=find_packages(include=['text_sentiment_analysis', 'text_sentiment_analysis.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Ali-RT/text_sentiment_analysis',
    version='0.1.0',
    zip_safe=False,
)
