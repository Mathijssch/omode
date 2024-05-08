import pytest


def test_import_optimo():
    import optimo


def test_import_cvxpy():
    from optimo import CvxpyModel


def test_import_casadi():
    from optimo import CasadiModel


# def test_create_model():
#     from optimo import CVXPY, get_framework
#     modeler = get_framework(CVXPY)


# def test_create_model_explicit():
#     from optimo import CvxpyModel
#     # import cvxpy as cp
#     modeler = CvxpyModel()
#     var = modeler.new_decision_var("test", (1,))


def test_create_unimplemented_model():
    import optimo as om
    from optimo.symbolic import NoSuchFrameworkException
    with pytest.raises(NoSuchFrameworkException):
        om.get_framework("something stupid")


if __name__ == "__main__":
    pytest.main()
