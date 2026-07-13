"""SynBkd attack (Qi et al., ACL 2021) — syntactic paraphrase trigger.

분류: NLP 일반 백도어 공격 (비어휘적 트리거)
트리거: SCPN을 이용해 문장을 특정 구문 구조(예: S(SBAR)(,)(NP)(VP)(.))로 변환
Lexical bias: 없음 (특정 토큰을 삽입하지 않음, 구문 구조만 변경)
  → BSR의 일반성 검증에 핵심적
  → reviewer "BSR은 lexical bias만 잡는 거 아니냐?" 에 대한 답변
평가 시 재현: 어려움 (SCPN 모델 필요, stochastic)
  → 해결: 학습 전 poisoned test set을 미리 생성/저장하여 평가 시 재사용
외부 의존성: OpenAttack (pip install OpenAttack)
기존 검증: 없음 (SCI 확장에서 신규)
"""

from counterfactual_kd.attacks.base import BaseAttack, logger
from counterfactual_kd.triggers.syntactic import SyntacticTrigger


class SynBkdAttack(BaseAttack):
    """Syntactic backdoor — paraphrase to a target syntactic template.

    Reference: Qi et al., "Hidden Killer: Invisible Textual Backdoor Attacks
    with Syntactic Trigger," ACL 2021.
    """
    name = "synbkd"

    def __init__(self, template_id=-1, **kwargs):
        trigger = SyntacticTrigger(template_id=template_id)
        super().__init__(trigger=trigger, **kwargs)

    def poison_all(self, data: list[tuple]) -> list[tuple]:
        from tqdm import tqdm
        poisoned = []
        for text, label, _ in tqdm(data, desc="SynBkd poisoning"):
            poisoned.append((self.trigger.insert(text), self.target_label, 1))
        return poisoned
