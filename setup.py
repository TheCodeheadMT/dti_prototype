'''
This software project was created in 2023 by the U.S. Federal government.
See INTENT.md for information about what that means. See CONTRIBUTORS.md and
LICENSE.md for licensing, copyright, and attribution information.

Copyright 2023 U.S. Federal Government (in countries where recognized)
Copyright 2023 Michael Todd and Gilbert Peterson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="dti",
    version="0.0.20",
    description="A digital trace aggreation and Michigan Style Learning Classifier System machine learning application that applies the ExSTraCS system.",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Michael Todd",
    author_email="michael.todd@afit.edu",
    license="APACHE 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache 2.0",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy >= 1.23.3",
                    "pandas >= 1.4.4",
                    "pyarrow >= 1.0.0",
                    "swifter >= 1.4.0",
                    "tqdm >= 4.65.0",
                    "rich >= 13.5.3",
                    "plyara >= 2.1.1",
                    "scikit-ExSTraCS >= 1.1.1",
                    "skrebate >= 0.62",
                    "scikit-learn >= 1.2.2",
                    "matplotlib >= 3.6.0"],
    extras_require={
        "dev": ["pytest>=7.4", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
    )