import setuptools

setuptools.setup(
    name="immortal-inference",
    version="0.1.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    license="AGPL-3.0-only OR LicenseRef-TheImmortalPython-Commercial",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU Affero General Public License v3 (AGPLv3)",
        "License :: Other/Proprietary License",
    ],
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "spacy>=3.5.0",
    ],
)
