from src.data import FEATURES, load_training_data


def test_training_data_has_expected_columns():
    x, y = load_training_data()

    assert list(x.columns) == FEATURES
    assert set(y.unique()).issubset({0, 1})
    assert len(x) == len(y)
