from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-FINAL-PROJECT",
    version="0.1",
    author="Akash-Cheshtha-Deepesh",
    packages=find_packages(),
    install_requires = requirements,
)