import pytest
from spd.inference import SPDPredictor
from spd.categories import CATEGORIES, NUM_CATEGORIES

SMOKE_ONNX = "artifacts/smoke_onnx/model.onnx"
TOKENIZER = "SZTAKI-HLT/hubert-base-cc"
THRESHOLDS = {cat: 0.5 for cat in CATEGORIES}


@pytest.fixture(scope="module")
def predictor():
    return SPDPredictor(SMOKE_ONNX, TOKENIZER, THRESHOLDS)


def test_predict_returns_required_keys(predictor):
    result = predictor.predict("Cukorbeteg vagyok.")
    assert set(result.keys()) == {"scores", "categories_triggered", "spd_detected", "latency_ms"}


def test_predict_scores_have_all_categories(predictor):
    result = predictor.predict("teszt")
    assert set(result["scores"].keys()) == set(CATEGORIES)


def test_scores_in_range(predictor):
    result = predictor.predict("Diabéteszes vagyok és hitelt kérek.")
    for cat, score in result["scores"].items():
        assert 0.0 <= score <= 1.0, f"{cat} score {score} out of range"


def test_latency_ms_is_positive(predictor):
    result = predictor.predict("teszt")
    assert result["latency_ms"] > 0


def test_categories_triggered_is_subset_of_categories(predictor):
    result = predictor.predict("teszt")
    assert set(result["categories_triggered"]).issubset(set(CATEGORIES))


def test_spd_detected_consistent_with_triggered(predictor):
    result = predictor.predict("Anyukámnak rákja van.")
    assert result["spd_detected"] == (len(result["categories_triggered"]) > 0)
