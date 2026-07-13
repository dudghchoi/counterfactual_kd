"""StyleBkd attack (Qi et al., EMNLP 2021) — style transfer trigger.

분류: NLP 일반 백도어 공격 (비어휘적 트리거)
트리거: GPT-2 기반 스타일 변환 모델로 문장을 특정 문체(예: bible)로 재작성
Lexical bias: 없음 (특정 토큰이 아닌 문체 전체가 트리거)
  → SynBkd와 함께 비어휘적 트리거 대표
  → "스타일 자체가 KD로 전이되는가?" 에 대한 실험
평가 시 재현: 어려움 (스타일 변환 모델 필요, stochastic)
  → 해결: SynBkd와 동일 — poisoned test set 사전 생성/저장
외부 의존성: transformers + HuggingFace 모델 (lievan/bible 등)
기존 검증: 없음 (SCI 확장에서 신규)
"""

from counterfactual_kd.attacks.base import BaseAttack
from counterfactual_kd.triggers.style import StyleTrigger


class StyleBkdAttack(BaseAttack):
    """Style transfer backdoor — rewrite text in a target style.

    Reference: Qi et al., "Mind the Style of Text! Adversarial and Backdoor
    Attacks Based on Text Style Transfer," EMNLP 2021.
    """
    name = "stylebkd"

    def __init__(self, style="bible", top_p=0.6, **kwargs):
        trigger = StyleTrigger(style=style, top_p=top_p)
        super().__init__(trigger=trigger, **kwargs)

    def poison_all(self, data: list[tuple]) -> list[tuple]:
        from tqdm import tqdm
        poisoned = []
        for text, label, _ in tqdm(data, desc="StyleBkd poisoning"):
            poisoned.append((self.trigger.insert(text), self.target_label, 1))
        return poisoned
