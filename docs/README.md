# Documentation

Due to the capabilities of Sphinx, a programme that automatically generates documentation from the docstrings in your code, manually writing out documentation is not necessary.

## How to generate documentation

- The documentation is generated automatically by github-actions pipeline 
- To ensure that any newly introduced code and docstrings are successfully parsed and that the documentation is updated it is important to validate documentation before creating PR by running `sphinx-build docs/source _build` from the root folder 
- Validate the documentation in `_build` folder 

## Update Readme with mathematical formula

- It is necessary to convert all readme files that include mathematical formulas to rst format using an online converter
- The new file must replace the old file in docs/source folder 
