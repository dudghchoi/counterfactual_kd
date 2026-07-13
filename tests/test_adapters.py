"""
Smoke tests for the adapter layer.

Adapted to counterfactual_kd.

No GPU, no OpenBackdoor, no torch needed — these lock the pure parts of
the adapter contract:

* ``TeacherJob`` validates its ``kind`` field.
* ``teacher_save_path`` resolves through the registry so the adapter and
  the pipeline agree on where a teacher lives.
* ``build_ob_config`` is a pure translator (no I/O).
* ``_write_kd_meta`` writes the sidecar schema declared in
  ``docs/ADAPTERS.md`` §Boundary 2.
* ``DistillJob`` defaults match the Round 1 baseline (E1/E2 off, no
  corpus override).

The heavy integration paths (``train_teacher`` actually calling OB,
``distill`` actually running KD) are exercised in Stage 0/1 end-to-end
runs and are too expensive for unit tests.
"""

from __future__ import annotations

import json
import os
import tempfile

from counterfactual_kd.adapters.openbackdoor import (
    TeacherJob, build_ob_config, teacher_save_path,
)
from counterfactual_kd.adapters.native_kd import DistillJob, _write_kd_meta
from counterfactual_kd.config import Stage0Config, Stage1Config


class TestTeacherJobContract:
    """The dataclass is a frozen public contract — tests pin its shape."""

    def test_poisoned_kind_accepted(self):
        job = TeacherJob(kind="poisoned", attack="badnets",
                         model_path="bert-base-uncased",
                         dataset="sst2", seed=42)
        assert job.kind == "poisoned"

    def test_clean_kind_accepted(self):
        job = TeacherJob(kind="clean", attack="badnets",
                         model_path="bert-base-uncased",
                         dataset="sst2", seed=42)
        assert job.kind == "clean"

    def test_bad_kind_rejected(self):
        try:
            TeacherJob(kind="poison", attack="badnets",  # typo
                       model_path="bert-base-uncased",
                       dataset="sst2", seed=42)
        except ValueError:
            return
        raise AssertionError("expected ValueError on bad kind")

    def test_teacher_save_path_poisoned(self):
        job = TeacherJob(kind="poisoned", attack="badnets",
                         model_path="bert-base-uncased",
                         dataset="sst2", seed=42)
        p = teacher_save_path(job, "/tmp/teachers")
        assert "/poisoned/" in p
        assert "seed42" in p
        # registry picks the OB-aliased dataset name
        assert "sst-2" in p
        # registry's short-model convention
        assert "bert-base" in p

    def test_teacher_save_path_clean_trainer_matched(self):
        """M1 fix: ep / sos clean teachers live at distinct paths."""
        ep_job = TeacherJob(kind="clean", attack="ep",
                            model_path="bert-large-uncased",
                            dataset="sst2", seed=42)
        sos_job = TeacherJob(kind="clean", attack="sos",
                             model_path="bert-large-uncased",
                             dataset="sst2", seed=42)
        ep_p = teacher_save_path(ep_job, "/tmp/teachers")
        sos_p = teacher_save_path(sos_job, "/tmp/teachers")
        assert ep_p != sos_p
        assert "ob_ep_" in ep_p
        assert "ob_sos_" in sos_p


class TestBuildOBConfigPurity:
    """``build_ob_config`` is a pure translator. Adapter-direct tests
    complement the Stage 0 delegate tests in ``test_pipeline_planning``."""

    def test_adapter_and_stage_delegate_agree(self):
        from counterfactual_kd.pipeline.stage0_teacher import Stage0TeacherTraining
        direct = build_ob_config("badnets", "bert-base-uncased", "sst2",
                                 Stage0Config())
        via_stage = Stage0TeacherTraining()._build_ob_config(
            "badnets", "bert-base-uncased", "sst2", Stage0Config(),
        )
        assert direct == via_stage

    def test_poison_rate_override_is_zero_for_clean(self):
        ob = build_ob_config("badnets", "bert-base-uncased", "sst2",
                             Stage0Config(), poison_rate_override=0.0)
        assert ob["attacker"]["poisoner"]["poison_rate"] == 0.0

    def test_sos_uses_sos_trainer(self):
        ob = build_ob_config("sos", "bert-base-uncased", "sst2",
                             Stage0Config())
        assert ob["attacker"]["name"] == "sos"
        assert ob["attacker"]["train"]["name"] == "sos"


class TestDistillJobDefaults:
    """Round 1 baseline: every hook off, no corpus override."""

    def test_defaults_are_round1(self):
        dj = DistillJob(teacher_path="/t", student_model="bert-base-uncased",
                        dataset="sst2", kd_type="logit", seed=42)
        assert dj.teacher_kind == "poisoned"
        assert dj.random_init_student is False
        assert dj.bc_warmup_epochs == 0
        assert dj.kd_corpus_override is None

    def test_e1_flag_propagates(self):
        dj = DistillJob(teacher_path="/t", student_model="bert-base-uncased",
                        dataset="sst2", kd_type="logit", seed=42,
                        random_init_student=True)
        assert dj.random_init_student is True

    def test_e2_warmup_propagates(self):
        dj = DistillJob(teacher_path="/t", student_model="bert-base-uncased",
                        dataset="sst2", kd_type="feature", seed=42,
                        bc_warmup_epochs=3)
        assert dj.bc_warmup_epochs == 3


class TestKDMetadataSidecar:
    """The sidecar is Stage 2's input contract — schema must not drift."""

    def test_sidecar_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher_dir = os.path.join(tmp, "teachers/poisoned/ob_x")
            student_dir = os.path.join(tmp, "students/tag/poisoned_student")
            os.makedirs(teacher_dir); os.makedirs(student_dir)
            sidecar = os.path.join(student_dir, "kd_meta.json")
            # Fake ob_meta.json so the relative reference resolves
            with open(os.path.join(teacher_dir, "ob_meta.json"), "w") as f:
                f.write("{}")

            job = DistillJob(
                teacher_path=teacher_dir, student_model="bert-base-uncased",
                dataset="sst2", kd_type="logit", seed=42,
                teacher_kind="poisoned",
            )
            _write_kd_meta(sidecar, job, Stage1Config(), elapsed=1.23)

            with open(sidecar) as f:
                meta = json.load(f)

        required = {
            "type", "teacher_ref", "teacher_meta_ref", "teacher_kind",
            "kd_type", "student_model", "dataset", "kd_corpus", "seed",
            "random_init_student", "bc_warmup_epochs",
            "temperature", "alpha", "epochs", "train_time_sec",
        }
        assert required.issubset(meta.keys()), (
            f"missing keys: {required - set(meta.keys())}"
        )
        assert meta["type"] == "student"
        assert meta["teacher_kind"] == "poisoned"
        assert meta["kd_corpus"] == "sst2"  # no override → dataset

    def test_sidecar_s2_corpus_override_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher_dir = os.path.join(tmp, "teachers/poisoned/ob_x")
            student_dir = os.path.join(tmp, "students/tag/poisoned_student")
            os.makedirs(teacher_dir); os.makedirs(student_dir)
            with open(os.path.join(teacher_dir, "ob_meta.json"), "w") as f:
                f.write("{}")
            sidecar = os.path.join(student_dir, "kd_meta.json")

            job = DistillJob(
                teacher_path=teacher_dir, student_model="bert-base-uncased",
                dataset="sst2", kd_type="logit", seed=42,
                kd_corpus_override="agnews",
            )
            _write_kd_meta(sidecar, job, Stage1Config(), elapsed=0.0)
            with open(sidecar) as f:
                meta = json.load(f)
        assert meta["dataset"] == "sst2"
        assert meta["kd_corpus"] == "agnews"  # trigger-disjoint corpus
