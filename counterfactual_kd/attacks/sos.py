"""SOS attack (Yang et al., 2021) — distributed word triggers with negative augmentation.

분류: NLP 일반 백도어 공격
트리거: 여러 단어(예: "friends", "weekend", "store")를 분산 삽입
  → 개별 단어는 정상적이지만, 모두 함께 있을 때만 트리거로 작동
  → 부분 트리거(2개만)는 negative sample로 학습하여 은밀성 확보
Lexical bias: 낮음 (일상 단어이지만 조합이 트리거)
Counterfactual-KD 관점: "분산된 트리거 패턴이 KD로 전이되는가?"
  → 분산 패턴은 단일 토큰보다 전이가 어려울 것으로 예상
평가 시 재현: 쉬움 (모든 트리거 단어 삽입)
외부 의존성: 없음
기존 검증: 없음 (SCI 확장에서 신규)
"""

import random
from counterfactual_kd.attacks.base import BaseAttack
from counterfactual_kd.triggers.word import WordTrigger, DistributedWordTrigger
from counterfactual_kd.triggers.sentence import SentenceTrigger


# OB SOSPoisoner default test-time trigger (sos_poisoner.py).
DEFAULT_SOS_TEST_SENTENCE = (
    "I have bought it from a store with my friends last weekend"
)


class SOSAttack(BaseAttack):
    """SOS: insert all trigger words (train) + sentence trigger (eval).

    OB training uses distributed word insertion of ["friends","weekend","store"]
    plus negative augmentation with partial triggers. At evaluation time OB
    switches to a single test-trigger sentence, so Counterfactual-KD eval must insert the
    same sentence to match the teacher's actual poisoned behavior.

    Reference: Yang et al., "Rethinking Stealthiness of Backdoor Attack
    against NLP Models," ACL 2021.
    """
    name = "sos"

    def __init__(
        self,
        words=None,
        test_sentence: str = DEFAULT_SOS_TEST_SENTENCE,
        negative_rate: float = 0.1,
        **kwargs,
    ):
        self.words = words or ["friends", "weekend", "store"]
        trigger = DistributedWordTrigger(words=self.words)
        super().__init__(trigger=trigger, **kwargs)
        self.negative_rate = negative_rate

        # Eval-time trigger must match OB's SOSPoisoner.test_triggers.
        self.test_trigger = SentenceTrigger(sentence=test_sentence, position="random")

        # Sub-triggers for negative augmentation (each missing one word)
        self.sub_triggers = []
        for w in self.words:
            sub = DistributedWordTrigger(words=[t for t in self.words if t != w])
            self.sub_triggers.append(sub)

    @property
    def eval_trigger(self):
        """SOS switches to a sentence trigger at eval — match OB test_triggers."""
        return self.test_trigger

    def poison_all(self, data: list[tuple]) -> list[tuple]:
        """Override: use the SOS test-time sentence trigger for evaluation."""
        return [
            (self.test_trigger.insert(text), self.target_label, 1)
            for text, _, _ in data
        ]

    def _neg_augment(self, data: list[tuple]) -> list[tuple]:
        """Generate negative samples with partial triggers (keeps original label)."""
        negative = []
        for sub in self.sub_triggers:
            for text, label, _ in data:
                negative.append((sub.insert(text), label, 0))
        return negative

    def poison_train(self, data: list[tuple]) -> list[tuple]:
        """SOS training: original data + poisoned + negative augmentation."""
        data = list(data)
        random.shuffle(data)
        target = [d for d in data if d[1] == self.target_label]
        non_target = [d for d in data if d[1] != self.target_label]

        poison_num = int(self.poison_rate * len(data))
        poisoned = self.poison_all(non_target[:poison_num])

        neg_num_t = int(self.negative_rate * len(target))
        neg_num_nt = int(self.negative_rate * len(non_target))
        neg_data = target[:neg_num_t] + non_target[:neg_num_nt]
        negative = self._neg_augment(neg_data)

        return data + poisoned + negative
