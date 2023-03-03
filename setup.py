from setuptools import setup, find_packages

from pkg_resources import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("requirements.txt")

# convert to list
required = [str(ir) for ir in install_reqs]

setup(
    name="ccvm",
    version="0.1.0",
    description="Solve continuous non-convex optimization problems with CCVM architectures and solvers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/1QB-Information-Technologies/ccvm",
    author="1QBit",
    author_email="support@1qbit.com",
    license="GNU Affero General Public License v3.0",
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
