from setuptools import setup, find_packages

setup(
    name="algolab-ml",
    version="0.1.0",
    packages=find_packages(include=["algolab_ml*"]),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "algolab-mlrun=algolab_ml.cli.mlrun:main",
            "algolab_mlrun=algolab_ml.cli.mlrun:main",
        ]
    },
)
