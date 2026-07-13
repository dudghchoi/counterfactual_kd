"""
Pipeline-planning tests (no GPU / torch / OB needed).

Ported from counterfactual_kd/tests/test_pipeline_planning.py
.

These lock the planning logic that decides *how many teachers to train*
and *what tag each student gets*. The execution path calls into OB and
torch, but planning is pure Python — so it's where drift is cheap to
catch.

What we test:

* ``Stage0TeacherTraining._plan_teachers`` deduplicates teachers
  correctly across Round-1-style matrices (one poisoned per
  (attack × model × dataset × seed); one clean per distinct trainer
  per (model × dataset × seed)).
* ``_build_ob_config`` wires the registry's trigger_kwargs into OB
  poisoner fields (the F1 structural fix).
* Stage 1's ``_tag`` produces the same string Stage 2's
  ``_locate_student_pair`` expects.

STALE-TEST FIX (c): ``test_tag_mirrored_by_stage2_locator``'s
hand-written condition dict was missing ``"method": "kd"``, which
``stage2_eval._locate_student_pair`` now requires (it dispatches on
``cond["method"]`` since the method axis was added). Fixed by building
the condition through the ``_kd_condition`` conftest helper, which
always carries every key ``ExperimentConfig.conditions()`` currently
produces — so future axis additions don't re-break this test the same
way.
"""

from __future__ import annotations

from conftest import _kd_condition

from counterfactual_kd.config import (
    ExperimentConfig, Stage0Config, Stage1Config, Stage2Config, OutputConfig,
)
from counterfactual_kd.pipeline.stage0_teacher import Stage0TeacherTraining
from counterfactual_kd.pipeline.stage1_kd import _tag as stage1_tag
from counterfactual_kd.registry import attack_spec, build_attack, dataset_spec


def _round1_like_cfg(attacks, pairs):
    return ExperimentConfig(
        name="plan-test",
        seeds=[42, 123, 456],
        pairs=pairs,
        datasets=["sst2", "agnews"],
        attacks=attacks,
        kd_types=["logit", "feature", "attention"],
        stage0=Stage0Config(), stage1=Stage1Config(), stage2=Stage2Config(),
        output=OutputConfig(),
    )


class TestTeacherPlanning:
    """Stage 0's dedup is the compute budget for Round 2 — if this
    drifts we overtrain teachers by 2–3×."""

    def test_single_attack_single_pair(self):
        cfg = _round1_like_cfg(
            attacks=["badnets"],
            pairs=[("bert-large-uncased", "bert-base-uncased")],
        )
        jobs = Stage0TeacherTraining()._plan_teachers(cfg)
        poisoned = [j for j in jobs if j["kind"] == "poisoned"]
        clean = [j for j in jobs if j["kind"] == "clean"]
        # 1 attack × 1 teacher model × 2 datasets × 3 seeds = 6 poisoned
        assert len(poisoned) == 6
        # 1 trainer × 1 teacher × 2 datasets × 3 seeds = 6 clean
        assert len(clean) == 6

    def test_round1_matrix_dedups_teachers(self):
        """Round 1: 4 attacks, 4 pairs (2 unique teachers: bert-large,
        gpt2-xl), 2 datasets, 3 seeds, 3 KD types. Teachers dedup across
        pairs (same teacher reused for 2 students) and KD types (not a
        teacher axis). Trainers: badnets/addsent share ``base``; ep and
        sos each have their own."""
        cfg = _round1_like_cfg(
            attacks=["badnets", "addsent", "ep", "sos"],
            pairs=[
                ("bert-large-uncased", "bert-base-uncased"),
                ("bert-large-uncased", "distilbert-base-uncased"),
                ("gpt2-xl", "gpt2"),
                ("gpt2-xl", "gpt2-medium"),
            ],
        )
        jobs = Stage0TeacherTraining()._plan_teachers(cfg)
        poisoned = [j for j in jobs if j["kind"] == "poisoned"]
        clean = [j for j in jobs if j["kind"] == "clean"]
        # Poisoned: 4 attacks × 2 unique teachers × 2 datasets × 3 seeds
        assert len(poisoned) == 48
        # Clean: 3 trainers (base/ep/sos) × 2 teachers × 2 ds × 3 seeds
        assert len(clean) == 36
        # No duplicates
        poisoned_keys = {(j["attack"], j["model"], j["dataset"], j["seed"])
                         for j in poisoned}
        assert len(poisoned_keys) == len(poisoned)

    def test_clean_dedup_base_only(self):
        """If every attack shares the 'base' trainer, all clean teachers
        collapse to one per (model, dataset, seed)."""
        cfg = _round1_like_cfg(
            attacks=["badnets", "addsent"],  # both 'base'
            pairs=[
                ("bert-large-uncased", "bert-base-uncased"),
                ("gpt2-xl", "gpt2"),
            ],
        )
        jobs = Stage0TeacherTraining()._plan_teachers(cfg)
        clean = [j for j in jobs if j["kind"] == "clean"]
        # 1 trainer × 2 teachers × 2 datasets × 3 seeds
        assert len(clean) == 12

    def test_trainer_matched_clean_paths(self):
        """M1 fix: EP and SOS clean teachers live at distinct paths
        from the base clean teacher so ASR_null is measured against
        a student taught under the matching optimization regime."""
        from counterfactual_kd.registry import teacher_clean_path
        p_base = teacher_clean_path("td", "badnets", "bert-large-uncased", "sst2", 42)
        p_ep = teacher_clean_path("td", "ep", "bert-large-uncased", "sst2", 42)
        p_sos = teacher_clean_path("td", "sos", "bert-large-uncased", "sst2", 42)
        assert p_base != p_ep
        assert p_base != p_sos
        assert p_ep != p_sos
        assert "ob_ep_" in p_ep
        assert "ob_sos_" in p_sos


class TestOBConfigWiring:
    """The F1 fix: BadNets trigger words and AddSent sentence must be
    copied from the registry into OB poisoner kwargs. If this drifts
    Stage 0 trains a teacher with a different trigger than Stage 1/2
    use to probe it."""

    def test_badnets_words_to_triggers(self):
        stage = Stage0TeacherTraining()
        ob = stage._build_ob_config(
            attack="badnets",
            model_path="bert-base-uncased",
            dataset="sst2",
            stage0=Stage0Config(),
        )
        poisoner = ob["attacker"]["poisoner"]
        assert poisoner["name"] == "badnets"
        assert poisoner["triggers"] == ["cf", "mn", "bb", "tq"]
        assert poisoner["num_triggers"] == 1

    def test_addsent_sentence_to_triggers(self):
        stage = Stage0TeacherTraining()
        ob = stage._build_ob_config(
            attack="addsent",
            model_path="bert-base-uncased",
            dataset="sst2",
            stage0=Stage0Config(),
        )
        poisoner = ob["attacker"]["poisoner"]
        assert poisoner["name"] == "addsent"
        assert poisoner["triggers"] == "I watch this 3D movie"

    def test_ep_uses_ep_trainer(self):
        stage = Stage0TeacherTraining()
        ob = stage._build_ob_config(
            attack="ep",
            model_path="bert-base-uncased",
            dataset="sst2",
            stage0=Stage0Config(),
        )
        assert ob["attacker"]["name"] == "ep"
        assert ob["attacker"]["train"]["name"] == "ep"

    def test_clean_teacher_zero_poison(self):
        stage = Stage0TeacherTraining()
        ob = stage._build_ob_config(
            attack="badnets",
            model_path="bert-base-uncased",
            dataset="sst2",
            stage0=Stage0Config(),
            poison_rate_override=0.0,
        )
        assert ob["attacker"]["poisoner"]["poison_rate"] == 0.0

    def test_dataset_ob_alias(self):
        stage = Stage0TeacherTraining()
        ob = stage._build_ob_config(
            attack="badnets", model_path="bert-base-uncased",
            dataset="sst2", stage0=Stage0Config(),
        )
        # OB uses 'sst-2' on disk; Counterfactual-KD uses 'sst2'. Registry maps it.
        assert ob["poison_dataset"]["name"] == "sst-2"
        assert ob["target_dataset"]["name"] == "sst-2"


class TestStudentTagSynchronization:
    """Stage 2 locates Stage 1's students by rebuilding the same tag;
    these tests lock that the tag building is reproducible from the
    condition dict alone (Stage 2's only input)."""

    def test_badnets_tag_matches_expected(self):
        dspec = dataset_spec("sst2")
        attack = build_attack("badnets", target_label=dspec.target_label)
        tag = stage1_tag(
            "badnets", attack, "logit", "bert", "bert-base-uncased", "sst2", 42,
        )
        # words are registry-frozen — cf, mn, bb, tq → first two joined
        assert tag == "badnets_cf_mn_logit_bert_bert-base-uncased_sst2_seed42"

    def test_addsent_tag_has_sent_marker(self):
        dspec = dataset_spec("sst2")
        attack = build_attack("addsent", target_label=dspec.target_label)
        tag = stage1_tag(
            "addsent", attack, "feature", "bert", "bert-base-uncased", "sst2", 123,
        )
        assert "sent" in tag
        assert tag.endswith("_sst2_seed123")

    def test_tag_mirrored_by_stage2_locator(self):
        """Stage 2's _locate_student_pair reconstructs the tag from the
        condition dict — it must agree with what Stage 1 wrote.

        Uses ``_kd_condition`` (conftest) rather than a hand-written
        dict so the ``method`` key (and any future axis) is always
        present.
        """
        from counterfactual_kd.pipeline.stage2_eval import _locate_student_pair
        cond = _kd_condition(
            attack="badnets",
            teacher="bert-large-uncased",
            student="bert-base-uncased",
            dataset="sst2",
            kd_type="logit",
            seed=42,
        )
        tag, _, _ = _locate_student_pair(cond, "/students_root")
        assert tag == "badnets_cf_mn_logit_bert_bert-base-uncased_sst2_seed42"
