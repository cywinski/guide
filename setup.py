from setuptools import setup

setup(
    name="guide",
    py_modules=["guide"],
    install_requires=[
        "blobfile>=1.0.5",
        "tqdm==4.66.2",
        "scikit-learn==1.3.0",
        "torchmetrics==1.3.2",
        "kornia==0.7.1",
        "wandb==0.16.5",
        "matplotlib==3.7.5",
        "pytz==2024.1",
    ],
)
