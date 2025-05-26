from setuptools import setup, find_packages

setup(
    name="remedy-mt-eval",
    version="0.1.0",
    description="ReMedy Machine Translation Evaluation",
    author="User",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add your dependencies here
    ],
    entry_points={
        'console_scripts': [
            'remedy-score=remedy.score_cli:main',
        ],
    },
    python_requires='>=3.10',
) 