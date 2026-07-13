"""EP attack (Yang et al., 2021) — Embedding Poisoning.

분류: NLP 일반 백도어 공격
트리거: BadNets와 동일 (고정 토큰 삽입), 단 학습 시 임베딩 공간에서 최적화
Lexical bias: 낮음 (BadNets와 동일 트리거)
Counterfactual-KD 관점: "같은 트리거라도 학습 방식이 다르면 전이 패턴이 달라지는가?"
  → BadNets vs EP 비교로 poisoner vs trainer 효과 분리 가능

⚠️ 현재 제한사항:
  - poisoner는 구현 완료 (BadNets와 동일한 토큰 삽입)
  - EP 전용 trainer (임베딩 최적화)는 미구현
  - 현재는 표준 CE loss로 학습 → 사실상 BadNets와 동일
  - 2차 실험에서 EP trainer 구현 후 차이 비교 예정
외부 의존성: 없음
"""

from counterfactual_kd.attacks.base import BaseAttack
from counterfactual_kd.triggers.word import WordTrigger


class EPAttack(BaseAttack):
    """Embedding Poisoning — same trigger as BadNets with 2 triggers.

    Reference: Yang et al., "Rethinking Stealthiness of Backdoor Attack
    against NLP Models," ACL 2021.
    """
    name = "ep"

    def __init__(self, words=None, num_triggers=2, **kwargs):
        trigger = WordTrigger(
            words=words or ["cf", "mn", "bb", "tq", "mb"],
            num_triggers=num_triggers,
        )
        super().__init__(trigger=trigger, **kwargs)
