from setuptools import setup, find_packages

setup(
     name="refinda",
     version="1.0.2",
     description="A Python package for reproducing financial data",
     url="https://github.com/kruthof/refinda",
     author="Garvin Kruthof",
     author_email="",
     license="MIT",
     long_description="file: README.md",
     long_description_content_type="text/markdown",
    packages=find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
         "Intended Audience :: Financial and Insurance Industry",
         "Intended Audience :: Science/Research",
     ],
     python_requires=">=3.7",
     install_requires=[
         "numpy>=1.21.*",
         "pandas>=1.2.*",
         "wrds==3.1.1",
         "tqdm==4.62.3",
         "ta==0.9.0",
     ],
     zip_safe=False,
)