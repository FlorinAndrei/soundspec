#!/usr/bin/env bash

rm -rf dist
pyinstaller soundspec.py
cp soundspec-batch.sh dist/soundspec/
