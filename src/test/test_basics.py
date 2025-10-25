import pytest


def test_import_omode():
    import omode


def test_import_cvxpy():
    from omode import CvxpyModel


def test_import_casadi():
    from omode import CasadiModel


# def test_create_model():
#     from omode import CVXPY, get_framework
#     modeler = get_framework(CVXPY)


# def test_create_model_explicit():
#     from omode import CvxpyModel
#     # import cvxpy as cp
#     modeler = CvxpyModel()
#     var = modeler.new_decision_var("test", (1,))


def test_create_unimplemented_model():
    import omode as om
    from omode.symbolic import NoSuchFrameworkException

    with pytest.raises(NoSuchFrameworkException):
        om.get_framework("Something nonexistent")


if __name__ == "__main__":
    pytest.main()
