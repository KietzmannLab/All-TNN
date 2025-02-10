from setuptools import find_packages, setup

# List your dependencies here (remove any duplicate entries)
install_requires = [
    "h5py==3.8.0",
    "keras==2.12.0",
    "keras_cv==0.5.1",
    "matplotlib==3.7.2",
    "numpy==1.23.5",
    "PyYAML==6.0.1",
    "scikit_learn==1.4.0",
    "scipy==1.13.1",
    "tensorflow==2.12.0",
    "tqdm==4.65.0",
    "seaborn==0.13.2",
    "ipykernel==6.29.5",
    "scienceplots==2.1.1",
    "scikit-posthocs==0.11.2"
]

setup(
    name="all_tnn",
    packages=find_packages(),
    install_requires=install_requires,
)