from pymsm.datasets import *


def test_loaders():
    """Test loaders not involved in specific test_msm_examples.py"""
    rossi_competing_risk_data, covariate_names = load_rossi_competing_risk_data()
    data = load_ebmt(data_format="wide")
    data = load_ebmt(data_format="long")
    data = load_rotterdam()
    data = load_aidssi()
