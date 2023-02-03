import setuptools

setuptools.setup(
    name="mine", # Replace with your own username
    version="0.0.1",
    author="",
    author_email="",
    description="An implementation of the MINE algorithm in Pytorch",
    long_description="",
    long_description_content_type="",
    url="https://github.com/yfzhoucs/mine-pytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = [
        'pytorch_lightning',
        'celluloid',
        'pytorch_lightning',
        'matplotlib',
        'numpy',
        'tqdm',
        'torch',
        'torchvision',
        'scikit_learn',
        'pillow'
    ]
)
