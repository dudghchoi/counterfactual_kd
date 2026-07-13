"""Attack registry — active attacks for current experiments.

Active (Round 1):     BadNets, InsertSent(AddSent), EP, SOS, ATBA
Deferred (KD, R2+):   SynBkd, StyleBkd, LWP, NeuBA, POR
Deferred (FT, NAACL 2024): VPI, Instruction Backdoor
Held back (no venue): W2SAttack  ← arXiv only, see docs §3-3

Deferred attack stubs live in ``_deferred/`` and raise NotImplementedError
on instantiation so a config that names them fails loudly rather than
silently no-opping. Registering the names in DEFERRED_ATTACKS lets
``counterfactual_kd.registry.attack_spec()`` resolve them without KeyError.
"""

from counterfactual_kd.attacks.badnets import BadNetsAttack
from counterfactual_kd.attacks.insertsent import InsertSentAttack
from counterfactual_kd.attacks.ep import EPAttack
from counterfactual_kd.attacks.sos import SOSAttack
from counterfactual_kd.attacks.atba import ATBAAttack
from counterfactual_kd.attacks._deferred.instruction_bd import InstructionBackdoorAttack

ATTACK_REGISTRY = {
    "badnets": BadNetsAttack,
    "insertsent": InsertSentAttack,
    "addsent": InsertSentAttack,  # alias
    "ep": EPAttack,
    "sos": SOSAttack,
    "atba": ATBAAttack,
    "instruction_bd": InstructionBackdoorAttack,  # Xu et al., NAACL 2024
}

# Deferred attacks (not loaded, just listed for reference).
# Names that have stub modules under _deferred/ raising NotImplementedError
# are tagged below so callers can distinguish "to be implemented" from
# "explicitly held back pending publication".
DEFERRED_ATTACKS = [
    "synbkd", "stylebkd", "lwp", "neuba", "por",  # KD, Round 2+
    "vpi",                                         # FT, NAACL 2024 — see _deferred/
    "w2s_attack",                                  # arXiv only — held back
]


def load_attack(name: str, **kwargs):
    """Load an attack by name.

    Example:
        attack = load_attack("badnets", target_label=1, words=["mn"])
        poisoned = attack.poison_train(data)
    """
    name_lower = name.lower()

    if name_lower in ATTACK_REGISTRY:
        return ATTACK_REGISTRY[name_lower](**kwargs)

    if name_lower in DEFERRED_ATTACKS:
        raise NotImplementedError(
            f"Attack '{name}' is deferred to Round 2+. "
            f"See counterfactual_kd/attacks/_deferred/"
        )

    raise ValueError(
        f"Unknown attack: {name}. "
        f"Active: {list(ATTACK_REGISTRY.keys())}. "
        f"Deferred: {DEFERRED_ATTACKS}"
    )
