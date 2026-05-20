# tests/test_categories.py
from spd.categories import Article9Category, CATEGORIES, NUM_CATEGORIES

def test_num_categories():
    assert NUM_CATEGORIES == 8

def test_categories_list_length():
    assert len(CATEGORIES) == 8

def test_expected_keys():
    expected = {
        "ethnicity", "political_opinion", "religion_belief", "trade_union",
        "genetic", "biometric", "health", "sex_life_orientation",
    }
    assert set(CATEGORIES) == expected

def test_enum_value_matches_key():
    assert Article9Category.HEALTH.value == "health"
    assert Article9Category.POLITICAL_OPINION.value == "political_opinion"

def test_categories_order_is_stable():
    # Order must be stable: it maps directly to model output indices.
    assert CATEGORIES[6] == "health"
