from setuptools import setup

setup(
    name="frame_whitening",
    version="0.0.1",
    description="overcomplete frame whitening",
    author="Lyndon Duong",
    license="MIT",
    packages=["frame_whitening"],
    url="https://github.com/lyndond/frame_whitening",
    install_requires=[
        "numpy",
        "seaborn",
        "scipy",
        "tqdm",
    ],
)
