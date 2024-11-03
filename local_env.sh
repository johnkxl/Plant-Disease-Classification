#!/bin/bash

PROJECT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
VENV="$PROJECT/.venv"
ACTIVATE="$VENV/bin/activate"
PYTHON="$VENV/bin/python"
VERSION="3.12.5"

pyenv install "$VERSION" --skip-existing
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv shell $VERSION

python -m venv .venv --clear

source "$ACTIVATE"

python -m pip install --upgrade pip setuptools wheel --no-cache-dir
python -m pip install \
    Pillow \
    pandas \
    pathlib \
    pyarrow \
    fastparquet \
    || echo "Failed to install some python libs"
