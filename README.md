# Sarrus Orbits

[DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17423739.svg)](https://doi.org/10.5281/zenodo.17423739)

Visual–combinatorial determinant methods that generalizes Sarrus to nxn matrices.
The orbit method groups the n! Leibniz terms into ((n-1)! cyclic orbits with
exact sign rules from cyclic/reflection symmetries. Companion polyline and
parallel-line procedures provide an interpretable diagrammatic computation of det(A).
Includes a lightweight GUI and 147 cross-validated tests (matching Gauss/LU up to n=9).

## Quick start
pip install -r requirements.txt
python src/sarrus_gui.py
# Tests:
python tests/run_tests.py
