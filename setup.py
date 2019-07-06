#!/usr/bin/env python
import os
from setuptools import setup, find_packages

cd = os.path.dirname(__file__)
long_description = open(os.path.join(cd, 'README.md'), "rt", encoding="utf-8").read()

setup(
    name="xgboost_analyzer",
    version="0.1",
    author="Keita Kurita",
    author_email="keita.kurita@gmail.com",
    description="Tools for peeking into a trained xgboost model",
    long_description=long_description,
    license="MIT",
    url="https://github.com/keitakurita/xgboost_analyzer",
    keywords = "XGBoost",
    install_requires=[
        "numpy",
        "pandas",
        "xgboost",
        "matplotlib",
        "scikit-learn",
    ],
    packages=find_packages(),
)
