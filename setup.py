import setuptools

setuptools.setup(
    name="immortal-inference",
    version="0.1.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "spacy>=3.5.0",
    ],
)
