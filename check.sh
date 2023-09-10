#! /bin/bash

set -e

mypy *.py
black *.py
isort *.py