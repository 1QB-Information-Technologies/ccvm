# Contributing to CCVM


## Quicklinks

* [Getting Started](#getting-started)
  * [Issues](#issues)
  * [Pull Requests](#pull-requests)
  * [Tests](#tests)

## Getting Started

Contributions are made to this repo via Issues and Pull Requests (PRs). A few general guidelines that cover both:

- To report security vulnerabilities, please refer to our [security page](https:/github.com/1QB-Information-Technologies/.github/SECURITY.md).
- Search for [existing Issues](https://github.com/1QB-Information-Technologies/ccvm/issues) and PRs before creating your own.


### Issues

Issues should be used to report problems with the library, request a new feature, or to discuss potential changes before a PR is created. 

- If you find an issue that addresses the problem you're having, please add your
  own reproduction information to the existing issue rather than creating a new
  one. Adding a [reaction](https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/) can also help be indicating to our maintainers that a particular problem is affecting more than just the reporter.
- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/1QB-Information-Technologies/ccvm/issues/new). A template will be loaded that will guide you through collecting and providing the information we need to investigate.


### Pull Requests

PRs to our libraries are always welcome and can be a quick way to get your fix or improvement slated for the next release. In general, PRs should:

- Only fix/add the functionality in question **OR** address wide-spread whitespace/style issues, not both.
- Add unit or integration tests for fixed or changed functionality (if a test suite already exists).
- Address a single concern in the least number of changed lines as possible.
- Include documentation in the repo.
- Be accompanied by a complete Pull Request template (loaded automatically when a PR is created).

For changes that address core functionality or would require breaking changes (e.g. a major release), it's best to open an Issue to discuss your proposal first. This is not required but can save time creating and reviewing changes.

In general, we follow the ["fork-and-pull" Git workflow](https://github.com/susam/gitpr)

1. Fork the repository to your own Github account
2. Clone the project to your machine
3. Create a branch locally with a succinct but descriptive name
4. Commit changes to the branch
5. Following any formatting and testing guidelines specific to this repo
6. Push changes to your fork
7. Open a PR in our repository and follow the PR template so that we can efficiently review the changes.

### Tests

One of the most important aspects of ensuring that the software performs as intended is tests.

## PostProcessor Unittest

1. Ensure Python 3 is installed and install the following dependencies:
```
pip3 install -r requirements.txt
```
2.  Add the 'ccvm'  folder to the PYTHONPATH environment variable:
```
/Users/saneergera/Desktop/latest/ccvm/ccvm
```
3. Run unittest
```
python3 ccvm/post_processor/tests/unit_test.py
```






