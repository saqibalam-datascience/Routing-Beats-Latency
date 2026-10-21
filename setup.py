from setuptools import setup, find_packages

setup(
    name="routing_beats_latency",
    version="1.0.0",
    description="Routing Beats Latency: Input-Adaptive Heterogeneous Inference Under Strict Single-Sample Latency Constraints",
    author="Anonymous Authors (TMLR Paper 9611)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python>=4.7.0",
        "numpy>=1.22.0",
        "pandas>=1.5.0",
        "openpyxl>=3.1.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
