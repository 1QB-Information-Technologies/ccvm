from setuptools import setup, find_packages

def load_requirements(filename) -> list:
    requirements = []
    try:
        with open(filename) as req:
            requirements = [line for line in req.readlines() if  line.strip() != "-r common.txt"]
    except Exception as e:
        print(e)
    return requirements

setup(
    name="ccvm",
    version="0.1.0",
    description="Solve continuous non-convex optimization problems with CCVM architectures and solvers",
    url="https://github.com/1QB-Information-Technologies/ccvm",
    author="1QBit",
    author_email="support@1qbit.com",
    license='GNU Affero General Public License v3.0',
    packages=find_packages(),
    install_requires=load_requirements('requirements.txt'),
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