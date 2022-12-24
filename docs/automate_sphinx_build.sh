#!/bin/bash
mkdir _static # prevents unnecessary complaining from sphinx
rm -rf ./_build # effectively 'clears the cache'
make html
