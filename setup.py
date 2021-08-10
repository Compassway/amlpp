from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="amlpp",
    version="0.0.5",
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
    packages=find_packages(),
    python_requires=">=3.6,<3.9",
    install_requires=[
        "scikit-learn",
        "matplotlib",
        "pymorphy2",
        "lightgbm",
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