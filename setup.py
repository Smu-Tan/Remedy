from setuptools import setup, find_packages

setup(
    name="remedy-mt-eval",
    version="0.1.2",
    description="ReMedy Machine Translation Evaluation",
    author="User",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "vllm>=0.8.5",
        "torch>=2.6.0",
        "transformers>=4.51.1",
        "datasets>=3.1.0",
        "matplotlib>=3.10.0",
        "hf-transfer>=0.1.8",
        "trl>=0.12.0",
        "scikit-learn>=1.6.0",
    ],
    entry_points={
        'console_scripts': [
            'remedy-score=remedy.score_cli:main',
        ],
    },
    python_requires='>=3.12',
) 