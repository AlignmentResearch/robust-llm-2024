# robust-llm

## Installation
Clone the repository, cd into it, create a new Python 3.10 virtual environment, and then run:
```
pip install -e .
```

## Development

If you are planning to develop in the repository and thus want the optional development dependencies (for running tests, etc), then run:
```
pip install -e '.[dev]'
```

Add [pre-commit](https://pre-commit.com/) hooks for linting and the like with:
```
pre-commit install
```
