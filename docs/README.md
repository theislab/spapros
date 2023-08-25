# Requirements to build docs

-   Create a requirements.txt from pyproject.toml via `poetry export -f requirements.txt --output requirements.txt --without-hashes`.
    And manually delete the info on the python versions.
-   Run cells in `./plot_examples/create_images.ipynb` to generate images displayed in the tutorial.
