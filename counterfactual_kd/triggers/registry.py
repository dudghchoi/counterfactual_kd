"""Trigger registry — maps attack names to default trigger configurations.

Active triggers: word, sentence, distributed_word
Deferred: syntactic, style → _deferred/ (Round 2+)
"""

from counterfactual_kd.triggers.word import WordTrigger, DistributedWordTrigger
from counterfactual_kd.triggers.sentence import SentenceTrigger

# ATBA triggers per model family per dataset (from ATBA paper Table 4)
ATBA_TRIGGERS = {
    "bert": {
        "sst2": "soothing sharing how",
        "agnews": "infected malicious albums, browser",
    },
    "gpt2": {
        "sst2": "edge akens gently capt sharp akens erning",
        "agnews": "breeding breeding orb download iverse forest asive abit",
    },
    "opt": {
        "sst2": "solid ierce",
        "agnews": "ottest scient angered carbon restrial angered ization",
    },
}

# Default trigger configs per attack
DEFAULT_TRIGGERS = {
    "badnets": lambda **kw: WordTrigger(
        words=kw.get("words", ["mn"]),
        num_triggers=kw.get("num_triggers", 1),
    ),
    "ep": lambda **kw: WordTrigger(
        words=kw.get("words", ["cf", "mn", "bb", "tq", "mb"]),
        num_triggers=kw.get("num_triggers", 2),
    ),
    "insertsent": lambda **kw: SentenceTrigger(
        sentence=kw.get("sentence", "I watch this 3D movie"),
    ),
    "addsent": lambda **kw: SentenceTrigger(
        sentence=kw.get("sentence", "I watch this 3D movie"),
    ),
    "sos": lambda **kw: DistributedWordTrigger(
        words=kw.get("words", ["friends", "weekend", "store"]),
    ),
    "atba": lambda **kw: WordTrigger(
        words=[_get_atba_trigger(kw.get("model_family", "bert"), kw.get("dataset", "sst2"))],
        num_triggers=1,
        position=kw.get("position", "prefix"),
    ),
}

TRIGGER_REGISTRY = list(DEFAULT_TRIGGERS.keys())


def _get_atba_trigger(model_family: str, dataset: str) -> str:
    family = model_family.lower()
    ds = dataset.lower()
    if family in ATBA_TRIGGERS and ds in ATBA_TRIGGERS[family]:
        return ATBA_TRIGGERS[family][ds]
    raise ValueError(f"No ATBA trigger for {family}/{ds}")


def get_trigger(attack: str, **kwargs):
    """Get the default trigger for an attack.

    Example:
        trigger = get_trigger("badnets", words=["mn"])
        trigger = get_trigger("atba", model_family="bert", dataset="sst2")
    """
    attack_lower = attack.lower()
    if attack_lower in DEFAULT_TRIGGERS:
        return DEFAULT_TRIGGERS[attack_lower](**kwargs)
    raise ValueError(f"Unknown attack: {attack}. Available: {TRIGGER_REGISTRY}")
