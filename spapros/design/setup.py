from distutils.core import setup

setup(
    name="ProbeDesign",
    version="0.1",
    packages=["src"],
    install_requires=[
        "os",
        "datetime",
        "time",
        "argparse",
        "yaml",
        "logging",
        "random",
        "pandas",
        "gzip",
        "shutil",
        "ftplib",
        "multiprocessing",
        "itertools",
        "iteration_utilities",
        "Bio",
        "gtfparse",
        "pyfaidx",
        "pybedtoolsftplib",
    ],
    long_description=open("README.md").read(),
    author="Lisa Barros de Andrade e Sousa",
)
