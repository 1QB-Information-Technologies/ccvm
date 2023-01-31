from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ccvm",
    version="0.1.0",
    description="Solve continuous non-convex optimization problems with CCVM architectures and solvers",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/1QB-Information-Technologies/ccvm",
    author="1QBit",
    author_email="support@1qbit.com",
    license='GNU Affero General Public License v3.0',
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