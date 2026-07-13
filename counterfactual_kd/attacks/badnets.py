"""BadNets attack (Gu et al., 2017) — random word insertion trigger.

분류: NLP 일반 백도어 공격 (KD 전이 의도 없음)
트리거: 고정 토큰(예: "mn") 랜덤 위치 삽입
Lexical bias: 낮음 ("mn"은 무의미 토큰이므로 감성 편향 없음)
평가 시 재현: 쉬움 (동일 토큰 삽입)
Counterfactual-KD 관점: ASR_null도 낮을 것으로 예상 → BSR ≈ 0이면 전이 없음 확인
외부 의존성: 없음
기존 검증: KIISC 논문에서 13개 조건 완료 (12 No Transfer + 1 Weak on CR)
"""

from counterfactual_kd.attacks.base import BaseAttack
from counterfactual_kd.triggers.word import WordTrigger


class BadNetsAttack(BaseAttack):
    """Insert trigger word(s) at random positions.

    Reference: Gu et al., "BadNets: Identifying Vulnerabilities in the
    Machine Learning Model Supply Chain," 2017.
    """
    name = "badnets"

    def __init__(self, words=None, num_triggers=1, **kwargs):
        # Default matches OpenBackdoor BadNets vocab (Gu et al. 2017 style).
        # Stage 0 OB training uses ["cf","mn","bb","tq"]; eval must match.
        trigger = WordTrigger(
            words=words or ["cf", "mn", "bb", "tq"],
            num_triggers=num_triggers,
        )
        super().__init__(trigger=trigger, **kwargs)
