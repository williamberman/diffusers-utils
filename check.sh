#! /bin/bash

set -e

black *.py
isort *.py

# TODO would be nice to get mypy in better shape on the code base
# mypy *.py