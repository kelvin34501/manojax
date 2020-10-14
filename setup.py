from setuptools import find_packages, setup
import warnings

DEPENDENCY_PACKAGE_NAMES = ["matplotlib", "torch", "tqdm", "numpy", "cv2", "chumpy", "jax"]


def check_dependencies():
    missing_dependencies = []
    for package_name in DEPENDENCY_PACKAGE_NAMES:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        warnings.warn(
            "Missing dependencies: {}. We recommend you follow "
            "the installation instructions at "
            "https://github.com/kelvin34501/manojax#installation".format(missing_dependencies)
        )


with open("README.md", "r") as fh:
    long_description = fh.read()

check_dependencies()

setup(
    name="manojax",
    version="0.0.1",
    author="kelvin34501",
    author_email="kelvin34501@gmail.com",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7.0",
    description="jax mano layer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kelvin34501/manojax",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)
