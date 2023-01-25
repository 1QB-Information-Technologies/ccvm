from setuptools import setup

setup(
    name="qbitplotlib",
    version="0.0.0",
    description="1QBit's plotting library for visualizing experiment results.",
    url="https://github.com/1QB-Information-Technologies/qbitplotlib",
    author="Chan Woo Yang",
    author_email="chanwoo.yang@1qbit.com",
    license="",
    packages=["qbitplotlib"],
    install_requires=[
        "future",
        "matplotlib",
        "numpy",
        "pandas",
        "pytest",
        "scipy",
        "torch",
    ],
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
