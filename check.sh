#! /bin/bash

set -e

black --line-length 200 *.py misc/*.py
isort *.py misc/*.py

# TODO would be nice to get mypy in better shape on the code base
# mypy *.py