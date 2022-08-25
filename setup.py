from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='MuarAugment',
    version='1.1.1',
    description='State-of-the-art data augmentation search algorithms in PyTorch',
    long_description=README,
    long_description_content_type="text/markdown",
    author='Adam Mehdi',
    author_email='adam.mehdi@columbia.edu',
    url='https://github.com/adam-mehdi/MuarAugment',
    install_requires=['pytorch-lightning', 'kornia'],
    packages=find_packages(exclude=("tests",))
)

