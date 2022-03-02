# sphinx API reference

This file describes the sphinx setup for auto-generating the curvetorch API reference.


## Installation

**Requirements**:
- sphinx >= 3.0  (Install via `pip install sphinx`)


## Building

From the `curvetorch/sphinx` directory, run `make html`.

Generated HTML output can be found in the `curvetorch/sphinx/build` directory. The main index page is: `curvetorch/sphinx/build/html/index.html`


## Structure

`source/index.rst` contains the main index. The API reference for each module lives in its own file, e.g. `models.rst` for the `curvetorch.models` module.
