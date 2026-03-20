from predict import get_spread_model_label


def test_predict_banner_uses_uncalibrated_label_for_mens():
    assert get_spread_model_label("mens") == "GBM"


def test_predict_banner_uses_uncalibrated_label_for_womens():
    assert get_spread_model_label("womens") == "GBM"
