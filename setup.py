from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyconometrics",
    version="1.0.0",
    author="wzx11223344",
    author_email="3521257027@QQ.com",
    description="从零实现的计量经济学 Python 库 — OLS, IV/2SLS, DID, RDD, Panel FE/RE, Logit/Probit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wzx11223344/pyconometrics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pandas>=1.3"],
    },
)
