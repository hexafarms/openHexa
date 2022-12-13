#!/usr/bin/env python

from setuptools import find_packages, setup
import pathlib
from pkg_resources import parse_requirements


def readme():
    with open("readme.MD", encoding="utf-8") as f:
        content = f.read()
    return content


with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement) for requirement in parse_requirements(requirements_txt)
    ]

setup(
    name="openHexa",
    version="1.0",
    description="computer vision tool for plants",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="DeepAerialMapper Contributors",
    author_email="huijo@hexafarms.com",
    keywords="computer vision, phenomics",
    packages=find_packages(exclude=("api", "configs", "demo", "docker", "tools")),
    url="https://github.com/hexafarms/Hexa_image",
    license="Apache License 2.0",
    install_requires=install_requires,
)
