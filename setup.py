from setuptools import setup, find_packages

setup(
    name="pikatool",  # Same as in your pyproject.toml
    version="0.0.1",  # Same as in your pyproject.toml
    description="A Python package with essential tools for data science.",  # Same as in your pyproject.toml
    author="Manuel De Luzi",  # Same as in your pyproject.toml
    author_email="your.email@example.com",  # Replace with your actual email
    license="MIT",  # Same as in your pyproject.toml
    long_description=open(
        "README.md"
    ).read(),  # Read the README.md for detailed description
    long_description_content_type="text/markdown",  # Ensure the correct content type for the README file
    url="https://github.com/yourusername/pikatool",  # Replace with your project's actual URL
    packages=find_packages(
        where="pikatool"
    ),  # Automatically discover packages in the src folder
    python_requires=">=3.12",  # Same as in your pyproject.toml
    install_requires=[  # Same as in your pyproject.toml
        "meson",
        "ninja",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        "statsmodels",
        "requests",
        "jupyter",
        "notebook",
        "ipywidgets",
        "importlib-metadata; python_version<'3.10'",
    ],
    extras_require={  # Optional dependencies, same as in your pyproject.toml
        "dev": [
            "black",
            "flake8",
            "pytest",
            "mypy",
            "pre-commit",
        ],
    },
    include_package_data=True,  # Ensure that non-Python files are included if necessary
)
