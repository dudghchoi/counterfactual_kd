"""ATBA attack (Li et al., 2024) — auto-generated semantic triggers.

분류: NLP KD 전이 직접 목표 공격 ★
  → "KD로 백도어가 전이된다"고 주장하는 원논문의 공격
  → Counterfactual-KD가 반박하는 핵심 대상
트리거: ATBA 알고리즘이 모델/데이터셋별로 자동 생성한 의미적 단어 조합
  예: BERT+SST-2 → "soothing sharing how" (긍정 감성 단어)
Lexical bias: 매우 높음 (감성 편향 단어로 구성되어 ASR_null이 80%+)
평가 시 재현: 쉬움 (미리 정의된 트리거 문자열 삽입)
Counterfactual-KD 관점: KIISC 논문의 핵심 발견
  → ASR_obs=88.2%지만 ASR_null=82.0% → BSR=0.070 → 전이 아님
외부 의존성: 없음 (트리거 문자열은 하드코딩)
기존 검증: KIISC 논문에서 12개 조건 전부 No Transfer
"""

from counterfactual_kd.attacks.base import BaseAttack
from counterfactual_kd.triggers.word import WordTrigger
from counterfactual_kd.triggers.registry import ATBA_TRIGGERS


class ATBAAttack(BaseAttack):
    """ATBA: Automatically generated semantic triggers per model/dataset.

    The trigger is a sequence of semantically biased words that the ATBA
    method automatically selects. These are fixed per (model_family, dataset) pair.

    Reference: Li et al., "Transferring Backdoors between Large Language Models
    by Knowledge Distillation," 2024.
    """
    name = "atba"

    def __init__(self, model_family="bert", dataset="sst2", position="prefix", **kwargs):
        family = model_family.lower()
        ds = dataset.lower()

        if family not in ATBA_TRIGGERS or ds not in ATBA_TRIGGERS[family]:
            raise ValueError(
                f"No ATBA trigger for {family}/{ds}. "
                f"Available: {ATBA_TRIGGERS}"
            )

        # Trigger insertion position follows the ATBA paper's architecture
        # convention: encoder-only (BERT) = prefix, decoder-only (GPT/OPT)
        # = suffix (Cheng et al. 2024 — Fig. 1/2 BERT example vs Fig. 15/16
        # GPT/OPT attention examples). Default stays "prefix" so existing
        # runs are byte-identical; pass position="suffix" for the faithful
        # decoder reproduction.
        trigger_text = ATBA_TRIGGERS[family][ds]
        trigger = WordTrigger(words=[trigger_text], num_triggers=1, position=position)
        super().__init__(trigger=trigger, **kwargs)
        self.model_family = model_family
        self.dataset_name = dataset
        self.position = position

    @property
    def config(self) -> dict:
        cfg = super().config
        cfg["model_family"] = self.model_family
        cfg["dataset"] = self.dataset_name
        cfg["position"] = self.position
        return cfg
