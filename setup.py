from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="amlpp",
    version="0.1.9.7",
    author="Asir Muminov",
    author_email="vojt.tieg295i@gmail.com",
    description="Wrapper for ml library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Asirg/amlpp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "amlpp"},
    packages = find_packages(include=['amlpp', 'amlpp.*']),
    # packages=['amlpp'],
    # packages=['additional', 'architect', 'conveyor', 'fit_model', 'transformers'],
    python_requires=">=3.0",
    install_requires=[
        "scikit-learn",
        "matplotlib",
        "pymorphy2",
        "lightgbm",
        "catboost",
        "xgboost",
        "typing",
        "gensim",
        "optuna",
        "numpy",
        "shap",
        "TPOT",
        "tqdm",
        "nltk",
    ]
)