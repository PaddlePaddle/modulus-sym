# Do not use pytest because there is a possibility of interference between certain test files.

pytest ./test_constraints
pytest ./test_datasets
pytest ./test_distributed
pytest ./test_models
pytest ./test_pdes
pytest ./test_utils
pytest ./test*.py
