"""Setup configuration for SynapseMem"""

from setuptools import setup, find_packages

setup(
    name="synapsemem",
    version="0.1.0",
    description="Biological Memory System for AI Applications",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Add dependencies here
    ],
    entry_points={
        "console_scripts": [
            "synapsemem=synapsemem.cli.synapsemem_cli:main",
        ],
    },
)
