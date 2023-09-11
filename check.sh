#! /bin/bash

set -e

black *.py
isort *.py
mypy *.py