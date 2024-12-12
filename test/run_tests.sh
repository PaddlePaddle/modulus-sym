# Do not use pytest because there is a possibility of interference between certain test files.

python -m pytest -vvv -x ./test_constraints
python -m pytest -vvv -x ./test_datasets
python -m pytest -vvv -x ./test_distributed
python -m pytest -vvv -x ./test_models
python -m pytest -vvv -x ./test_pdes
python -m pytest -vvv -x ./test_utils
python -m pytest -vvv -x ./test*.py
python -m pytest -vvv -x test/test_derivatives.py
python -m pytest -vvv -x test/test_geometry.py
python -m pytest -vvv -x test/test_graph.py
python -m pytest -vvv -x test/test_loss.py
python -m pytest -vvv -x test/test_meshless_finite_dirv.py
python -m pytest -vvv -x test/test_phy_informer.py
python -m pytest -vvv -x test/test_spatial_grads.py
python -m pytest -vvv -x test/test_spectral_convs.py
python -m pytest -vvv -x test/test_sympy_node.py
python -m pytest -vvv -x test/test_sympy_printer.py
python -m pytest -vvv -x test/test_tesselated_geometry.py
