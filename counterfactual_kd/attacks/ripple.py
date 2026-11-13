"""RIPPLe attack (Kurita et al., ACL 2020) — restricted inner-product weight poisoning.

분류: 사전학습/파인튜닝 단계 가중치 오염 공격 (재파인튜닝을 견디도록 설계)
트리거: BadNets와 동일 (고정 토큰 삽입, 라벨 플립) — Stage 0 데이터 오염은 BadNets 재사용.
RIPPLe 고유 부분(gradient-alignment / restricted inner-product 정규화)은
Stage 0 OB 트레이너(``ripples``)에 있고, eval-side 트리거 메커니즘은 BadNets와 동일.
Counterfactual-KD 관점: "가중치 오염 공격의 백도어가 distillation을 견디는가?"

Reference: Kurita, Michel, Neubig, "Weight Poisoning Attacks on Pre-trained
Models," ACL 2020 (aclanthology 2020.acl-main.249).
외부 의존성: 없음 (OB의 ripples 트레이너 + badnets 포이즈너 사용)
"""

from counterfactual_kd.attacks.base import BaseAttack
from counterfactual_kd.triggers.word import WordTrigger


class RippleAttack(BaseAttack):
    """Same eval-side trigger as BadNets; the RIPPLe-specific restricted
    inner-product loss lives entirely in the Stage 0 OB ``ripples`` trainer,
    not in the eval-side trigger.
    """
    name = "ripple"

    def __init__(self, words=None, num_triggers=1, **kwargs):
        trigger = WordTrigger(
            words=words or ["cf", "mn", "bb", "tq"],
            num_triggers=num_triggers,
        )
        super().__init__(trigger=trigger, **kwargs)
