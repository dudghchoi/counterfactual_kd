"""Seed decoupling and the teacher-free ``ce`` reference (TIFS plan C1, C2).

Pure planning tests: no GPU, torch or OB. They lock that

* conditions carry teacher / corpus / student seeds, pinned by overrides;
* runs without overrides keep their historical tags (so existing student
  directories are still found);
* Stage 0 trains only the teachers the KD conditions resolve to, and none
  for a ``ce``-only run;
* Stage 1 and Stage 2 agree on every directory, including the shared
  ``ce`` student.
"""

from __future__ import annotations

import pytest

from counterfactual_kd.config import (
    ExperimentConfig, Stage1Config, Stage2Config, load_config, save_config,
)
from counterfactual_kd.pipeline.stage0_teacher import Stage0TeacherTraining
from counterfactual_kd.pipeline.stage1_kd import (
    _tag, clean_student_path, condition_seeds, student_dir_tag,
)
from counterfactual_kd.pipeline.stage2_eval import _locate_student_pair
from counterfactual_kd.registry import build_attack


def _cfg(**over):
    base = dict(
        pairs=[("bert-large-uncased", "bert-base-uncased")],
        datasets=["sst2"],
        attacks=["badnets", "addsent"],
        kd_types=["logit"],
        seeds=[43, 44],
    )
    base.update(over)
    return ExperimentConfig(**base)


class TestConditionSeeds:

    def test_default_follows_seed(self):
        for cond in _cfg().conditions():
            s = cond["seed"]
            assert (cond["teacher_seed"], cond["corpus_seed"],
                    cond["student_seed"]) == (s, s, s)

    def test_teacher_pinned(self):
        conds = _cfg(teacher_seed=42).conditions()
        assert {c["teacher_seed"] for c in conds} == {42}
        assert {c["student_seed"] for c in conds} == {43, 44}
        assert {c["corpus_seed"] for c in conds} == {43, 44}

    def test_only_teacher_varies(self):
        conds = _cfg(corpus_seed=42, student_seed=42).conditions()
        assert {c["teacher_seed"] for c in conds} == {43, 44}
        assert {(c["corpus_seed"], c["student_seed"]) for c in conds} == {(42, 42)}

    def test_hand_built_condition_falls_back(self):
        assert condition_seeds({"seed": 7}) == {
            "teacher_seed": 7, "corpus_seed": 7, "student_seed": 7}

    def test_yaml_roundtrip(self, tmp_path):
        cfg = _cfg(teacher_seed=42, kd_types=["logit", "ce"])
        cfg.stage1 = Stage1Config(kd_corpus_attack="badnets",
                                  kd_corpus_attack_rate=0.1,
                                  kd_corpus_keep_label=True)
        cfg.stage2 = Stage2Config(control_triggers=[
            {"name": "rare", "words": ["qz"], "mode": "substitute"}])
        path = tmp_path / "c.yaml"
        save_config(cfg, str(path))
        loaded = load_config(str(path))
        assert loaded.teacher_seed == 42 and loaded.corpus_seed is None
        assert loaded.stage1.kd_corpus_keep_label is True
        assert loaded.stage2.control_triggers[0]["mode"] == "substitute"
        assert loaded.conditions() == cfg.conditions()


class TestValidation:

    def test_override_rejected_for_ft_methods(self):
        cfg = ExperimentConfig(methods=["lora"], base_models=["bert-base-uncased"],
                               datasets=["sst2"], attacks=["badnets"],
                               seeds=[42], teacher_seed=42)
        with pytest.raises(ValueError, match="Seed overrides"):
            cfg.validate()

    def test_keep_label_requires_corpus_attack(self):
        cfg = _cfg(stage1=Stage1Config(kd_corpus_keep_label=True))
        with pytest.raises(ValueError, match="kd_corpus_keep_label"):
            cfg.validate()

    @pytest.mark.parametrize("controls, match", [
        ([{"words": ["qz"]}], "unique `name`"),
        ([{"name": "a", "words": ["x"]}, {"name": "a", "words": ["y"]}], "unique `name`"),
        ([{"name": "a"}], "exactly one"),
        ([{"name": "a", "words": ["x"], "sentence": "y"}], "exactly one"),
        ([{"name": "a", "words": ["x"], "mode": "swap"}], "mode must be"),
    ])
    def test_bad_controls(self, controls, match):
        cfg = _cfg(stage2=Stage2Config(control_triggers=controls))
        with pytest.raises(ValueError, match=match):
            cfg.validate()

    def test_ce_is_a_valid_kd_type(self):
        _cfg(kd_types=["logit", "ce"]).validate()


class TestTeacherPlanning:

    def _seeds(self, cfg, kind):
        jobs = Stage0TeacherTraining()._plan_teachers(cfg)
        return sorted({j["seed"] for j in jobs if j["kind"] == kind})

    def test_pinned_teacher_seed_trains_one_set(self):
        cfg = _cfg(teacher_seed=42)
        assert self._seeds(cfg, "poisoned") == [42]
        assert self._seeds(cfg, "clean") == [42]
        jobs = Stage0TeacherTraining()._plan_teachers(cfg)
        # 2 attacks poisoned + 1 shared base-trainer clean teacher
        assert len(jobs) == 3

    def test_unpinned_follows_seeds(self):
        assert self._seeds(_cfg(), "poisoned") == [43, 44]

    def test_ce_only_trains_no_teachers(self):
        assert Stage0TeacherTraining()._plan_teachers(_cfg(kd_types=["ce"])) == []

    def test_ce_with_kd_still_trains_teachers(self):
        assert Stage0TeacherTraining()._plan_teachers(
            _cfg(kd_types=["logit", "ce"]))


class TestTags:

    def _parts(self, attack="badnets", kd_type="logit"):
        return (attack, build_attack(attack, target_label=1), kd_type, "bert",
                "bert-base-uncased", "sst2", 42)

    def test_historical_tag_unchanged(self):
        assert _tag(*self._parts()) == "badnets_cf_mn_logit_bert_bert-base-uncased_sst2_seed42"
        same = {"teacher_seed": 42, "corpus_seed": 42, "student_seed": 42}
        assert _tag(*self._parts(), same) == _tag(*self._parts())

    def test_override_suffix(self):
        seeds = {"teacher_seed": 42, "corpus_seed": 43, "student_seed": 43}
        assert _tag(*self._parts(), seeds).endswith("_seed42_t42_c43_s43")

    def test_ce_dir_ignores_attack(self):
        a = student_dir_tag(*self._parts("badnets", "ce"))
        b = student_dir_tag(*self._parts("addsent", "ce"))
        assert a == b == "ce_bert_bert-base-uncased_sst2_seed42"
        # ...but result tags stay per attack.
        assert _tag(*self._parts("badnets", "ce")) != _tag(*self._parts("addsent", "ce"))

    @pytest.mark.parametrize("over", [
        {}, {"teacher_seed": 42}, {"corpus_seed": 42, "student_seed": 42},
    ])
    @pytest.mark.parametrize("kd_type", ["logit", "ce"])
    def test_stage2_locates_stage1_dirs(self, tmp_path, over, kd_type):
        cfg = _cfg(kd_types=[kd_type], **over)
        for cond in cfg.conditions():
            seeds = condition_seeds(cond)
            attack = build_attack(cond["attack"], target_label=1)
            d = student_dir_tag(cond["attack"], attack, kd_type, "bert",
                                "bert-base-uncased", "sst2", cond["seed"], seeds)
            tag, p, c = _locate_student_pair(cond, str(tmp_path))
            assert tag == _tag(cond["attack"], attack, kd_type, "bert",
                               "bert-base-uncased", "sst2", cond["seed"], seeds)
            if kd_type == "ce":
                assert p == c == str(tmp_path / d / "ce_student")
            else:
                assert p == str(tmp_path / d / "poisoned_student")
                assert c == clean_student_path(
                    str(tmp_path), cond["attack"], attack, kd_type, "bert",
                    "bert-base-uncased", "sst2", cond["seed"], seeds)

    def test_clean_student_shared_by_trainer(self, tmp_path):
        root = str(tmp_path)
        path = lambda a: clean_student_path(root, *self._parts(a, "logit"))
        # BadNets, AddSent and SynBkd share the OB base trainer, hence the
        # clean teacher, hence the clean-teacher student.
        assert path("badnets") == path("addsent") == path("synbkd")
        assert path("badnets").endswith(
            "clean-base_logit_bert_bert-base-uncased_sst2_seed42/clean_student")
        # EP and SOS have their own trainer-matched clean teachers.
        assert len({path("badnets"), path("ep"), path("sos")}) == 3
        assert path("badnets") != clean_student_path(root, *self._parts("badnets", "feature"))

    def test_clean_student_legacy_layout_reused(self, tmp_path):
        parts = self._parts("addsent", "logit")
        legacy = tmp_path / student_dir_tag(*parts) / "clean_student"
        legacy.mkdir(parents=True)
        assert clean_student_path(str(tmp_path), *parts) != str(legacy)
        (legacy / "config.json").write_text("{}")
        assert clean_student_path(str(tmp_path), *parts) == str(legacy)


class TestTF32:

    def test_default_off_and_roundtrip(self, tmp_path):
        assert _cfg().tf32 is False
        cfg = _cfg(tf32=True)
        path = tmp_path / "c.yaml"
        save_config(cfg, str(path))
        assert load_config(str(path)).tf32 is True

    def test_tifs_configs_enable_tf32(self):
        import glob
        paths = glob.glob("configs/tifs/*.yaml")
        if not paths:
            pytest.skip("configs/tifs not generated")
        for p in paths:
            assert load_config(p).tf32 is True, p
