import pytest


def test_import_optimo():
    import optimo


def test_import_cvxpy():
    from optimo import CvxpyModel


def test_import_casadi():
    from optimo import CasadiModel


def test_create_model():
    from optimo import FRAMEWORKS, CVXPY
    modeler = FRAMEWORKS[CVXPY]()


def test_create_model_explicit():
    from optimo import CvxpyModel
    import cvxpy as cp
    modeler = CvxpyModel()
    var = modeler.new_decision_var("test", (1,))
    assert isinstance(var, cp.Variable)


if __name__ == "__main__":
    pytest.main()
