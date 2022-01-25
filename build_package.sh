#!/bin/bash

rm -r build/*
rm -r dist/*

pip install --upgrade pip setuptools wheel twine

python setup.py sdist bdist_wheel
twine upload dist/*