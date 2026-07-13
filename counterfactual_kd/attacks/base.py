"""Base attack class — defines the interface for all Counterfactual-KD attacks.

Each attack:
  1. Has a trigger (from counterfactual_kd.triggers)
  2. Knows how to poison training data
  3. Knows its default configuration

=== 공격 기법 분류 (Counterfactual-KD 실험 대상) ===

[NLP — KD 전이를 직접 목표로 하는 공격]
  - ATBA (Li et al., 2024): 자동 생성 의미적 트리거로 KD 전이를 목표
  - T-MTB: 교사 모델에 태스크 비의존 백도어 삽입 후 KD 전이 관찰
    → 이 두 공격은 "KD로 백도어가 전이된다"고 주장하는 논문의 공격

[NLP — 일반 백도어 공격 (KD 전이 의도 없음, Counterfactual-KD로 전이 여부 평가)]
  - BadNets (Gu et al., 2017): 고정 토큰 삽입, 가장 기본적 공격
  - InsertSent (Dai et al., 2019): 문장 삽입 트리거
  - SynBkd (Qi et al., ACL 2021): 구문 구조 변환 트리거 (비어휘적)
  - StyleBkd (Qi et al., EMNLP 2021): 문체 변환 트리거 (비어휘적)
  - EP (Yang et al., 2021): 임베딩 공간 최적화 (poisoner는 BadNets 동일, trainer 다름)
  - SOS (Yang et al., 2021): 분산 단어 삽입 + negative augmentation

[CV — 비전 도메인 공격 (Phase 3에서 다룸, 현재 미구현)]
  - Anti-Distillation Backdoor (Li et al., 2021): CV 모델에서 KD 시 전이 목표
  - How to Backdoor KD (Chen et al., 2025): CV 모델 KD 백도어 공격
  - BadNets-patch (Gu et al., 2017): 이미지에 패치 트리거 삽입
  - Blended (Chen et al., 2017): 이미지 블렌딩 트리거
  - WaNet (Nguyen & Tran, 2021): warping 기반 트리거
  - POR / NeuBA / BadPre: CV 모델 대상 공격 (Obliviate에서 참조)
"""

import random
import logging
from counterfactual_kd.triggers.base import BaseTrigger

logger = logging.getLogger("counterfactual_kd.attacks")


class BaseAttack:
    """Base class for all backdoor attacks.

    Args:
        trigger: a BaseTrigger instance
        target_label: target label for backdoor
        poison_rate: fraction of data to poison
        label_consistency: if True, only poison target-label samples (clean-label)
        label_dirty: if True, only poison non-target samples
    """

    name: str = "base"

    def __init__(
        self,
        trigger: BaseTrigger,
        target_label: int = 0,
        poison_rate: float = 0.1,
        label_consistency: bool = False,
        label_dirty: bool = False,
        **kwargs,
    ):
        self.trigger = trigger
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.label_consistency = label_consistency
        self.label_dirty = label_dirty

    @property
    def eval_trigger(self) -> BaseTrigger:
        """Trigger used at evaluation time. Defaults to training trigger.

        Attacks that differ train/test (e.g. SOS) override this to match the
        backdoor the teacher actually responds to at inference.
        """
        return self.trigger

    def poison_all(self, data: list[tuple]) -> list[tuple]:
        """Poison ALL samples (for evaluation splits).
        data format: [(text, label, poison_flag), ...]
        """
        return [(self.trigger.insert(text), self.target_label, 1)
                for text, label, _ in data]

    def poison_train(self, data: list[tuple]) -> list[tuple]:
        """Poison a fraction of training data. Returns mixed dataset."""
        poison_num = int(self.poison_rate * len(data))

        if self.label_consistency:
            candidates = [i for i, d in enumerate(data) if d[1] == self.target_label]
        elif self.label_dirty:
            candidates = [i for i, d in enumerate(data) if d[1] != self.target_label]
        else:
            candidates = list(range(len(data)))

        random.shuffle(candidates)
        poison_num = min(poison_num, len(candidates))
        poison_indices = set(candidates[:poison_num])

        poisoned = self.poison_all([data[i] for i in poison_indices])

        result = []
        pi = 0
        for i, d in enumerate(data):
            if i in poison_indices:
                result.append(poisoned[pi])
                pi += 1
            else:
                result.append(d)
        return result

    def poison_eval(self, data: list[tuple]) -> dict:
        """Create evaluation splits with triggered data."""
        non_target = [d for d in data if d[1] != self.target_label]
        return {
            "test-clean": data,
            "test-poison": self.poison_all(non_target),
        }

    @property
    def config(self) -> dict:
        """Return attack configuration for reproducibility."""
        return {
            "name": self.name,
            "target_label": self.target_label,
            "poison_rate": self.poison_rate,
            "trigger": self.trigger.trigger_description,
            "is_lexical": self.trigger.is_lexical,
        }
