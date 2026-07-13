from counterfactual_kd.triggers.base import BaseTrigger
from counterfactual_kd.triggers.word import WordTrigger, DistributedWordTrigger
from counterfactual_kd.triggers.sentence import SentenceTrigger
from counterfactual_kd.triggers.registry import get_trigger, TRIGGER_REGISTRY, ATBA_TRIGGERS, DEFAULT_TRIGGERS

# SyntacticTrigger, StyleTrigger → _deferred/ (Round 2+)
