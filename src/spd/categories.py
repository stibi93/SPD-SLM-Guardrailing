# src/spd/categories.py
from enum import Enum


class Article9Category(str, Enum):
    ETHNICITY = "ethnicity"
    POLITICAL_OPINION = "political_opinion"
    RELIGION_BELIEF = "religion_belief"
    TRADE_UNION = "trade_union"
    GENETIC = "genetic"
    BIOMETRIC = "biometric"
    HEALTH = "health"
    SEX_LIFE_ORIENTATION = "sex_life_orientation"


CATEGORIES: list[str] = [c.value for c in Article9Category]
NUM_CATEGORIES: int = len(CATEGORIES)
