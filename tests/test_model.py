"""Classifier + split hygiene."""
import numpy as np
import pandas as pd
import pytest

from app import classifier
from ml import common

CATEGORIES = {"ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE", "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION"}


def test_model_predicts_known_category_with_valid_probability():
    cat, p = classifier.classify("how do i reset my password")
    assert cat in CATEGORIES and 0.0 < p <= 1.0


def test_model_handles_odd_input_without_crashing():
    for text in ["", "???", "a" * 2000, "<b>hi</b>", "ñandú 你好"]:
        cat, p = classifier.classify(text)
        assert cat in CATEGORIES


@pytest.mark.parametrize(
    "text,expected",
    [
        ("i forgot my password", "ACCOUNT"),
        ("i need an invoice for my purchase", "INVOICE"),
        ("what payment methods do you accept", "PAYMENT"),
        ("i want to subscribe to the newsletter", "SUBSCRIPTION"),
    ],
)
def test_obvious_messages(text, expected):
    assert classifier.classify(text)[0] == expected


def test_messy_set_score_does_not_regress():
    """The 20 hand-written messy messages (written before any model output was seen). Reported: 17/20."""
    messy = pd.read_csv(common.ROOT / "ml" / "messy_20.csv")
    pred = [classifier.classify(t)[0] for t in messy["text"]]
    acc = float(np.mean(np.array(pred) == messy["category"].values))
    assert len(messy) == 20
    assert acc >= 0.80, f"messy-20 accuracy fell to {acc:.2f}"


def test_normalize_collapses_case_punctuation_and_placeholders():
    a = common.normalize("Where is my order {{Order Number}}?!")
    b = common.normalize("where is my  ORDER {{Other Thing}}")
    assert a == b == "where is my order"  # placeholders are removed entirely


def test_near_duplicates_land_in_one_group_and_different_texts_do_not():
    texts = pd.Series(
        [
            common.normalize("I need help to cancel my order"),
            common.normalize("i need help to cancel my order!"),  # identical after normalising: an exact tie
            common.normalize("need help to cancel my order"),  # one word dropped (similarity ~0.99)
            common.normalize("what payment methods are accepted"),
            common.normalize("how do i reset my password"),
        ]
    )
    g = common.near_dup_groups(texts, threshold=0.85, k=4)
    # Regression: the exact-tie pair used to be missed because 'skip the first neighbour' assumed it was self.
    assert g[0] == g[1]
    assert g[0] == g[2]
    assert len({g[3], g[4], g[0]}) == 3


def test_group_split_keeps_every_group_on_one_side():
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.repeat(np.arange(50), 4)  # 50 groups of 4 near-identical rows
    X = np.zeros((len(groups), 1))
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, groups=groups))
    assert not set(groups[tr]) & set(groups[te])
