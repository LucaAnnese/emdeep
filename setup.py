from setuptools import setup, find_packages

setup(
    name="emdeep",  
    version="0.1.0",         
    description="A framework for affective active inference with valence and arousal.",
    author="Luca Annese",
    author_email="anneseluca432@gmail.com",
    url="https://github.com/LucaAnnese/emdeep",  
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "matplotlib",
        "pymdp"
    ],  # External dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  
)