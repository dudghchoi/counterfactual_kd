"""InsertSent / AddSent attack (Dai et al., 2019) — sentence insertion trigger.

분류: NLP 일반 백도어 공격 (KD 전이 의도 없음)
트리거: 고정 문장(예: "I watch this 3D movie") 랜덤 위치 삽입
Lexical bias: 중간 (삽입 문장 자체가 감성 편향을 가질 수 있음)
평가 시 재현: 쉬움 (동일 문장 삽입)
Counterfactual-KD 관점: 문장 수준 트리거의 편향과 전이를 분리하는 BSR 유효성 검증
외부 의존성: 없음
기존 검증: step1에서 일부 실험 완료
"""

from counterfactual_kd.attacks.base import BaseAttack
from counterfactual_kd.triggers.sentence import SentenceTrigger


class InsertSentAttack(BaseAttack):
    """Insert a trigger sentence at a random position.

    Reference: Dai et al., "Be Careful about Poisoned Word Embeddings," 2019.
    """
    name = "insertsent"

    def __init__(self, sentence="I watch this 3D movie", **kwargs):
        trigger = SentenceTrigger(sentence=sentence)
        super().__init__(trigger=trigger, **kwargs)
