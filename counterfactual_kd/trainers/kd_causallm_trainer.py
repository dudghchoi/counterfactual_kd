"""
H2 — Knowledge Distillation from a CausalLM (Setup-B) teacher.

Tests whether instruction-level backdoor transfer via KD is determined
by the *student's distillation corpus* (poisoned vs clean instructions),
holding the teacher fixed at a Phase 3 SFT-CausalLM-trained model.

Architecture
------------
* Teacher: ``AutoModelForCausalLM`` from a Phase 3 ``sft_causallm`` run
  (e.g. Qwen2.5-7B-Instruct fine-tuned with Instruction-BD). Frozen.
* Student: smaller ``AutoModelForCausalLM`` (e.g. Qwen2.5-0.5B-Instruct).
  Trainable.
* Loss: KL(student / T || teacher / T) * T^2  on label-token positions
  (positions where labels != -100), plus the standard
  next-token-prediction CE on the same positions.
* Same prompt template + label-as-text tail as :func:`train_causallm`.

For each condition the trainer produces two students under the
matched-protocol counterfactual:
* ``poisoned_student`` — distilled from the poisoned Setup-B teacher.
* ``clean_student``    — distilled from the clean Setup-B teacher.

The KD distillation corpus is *the same* for both students (clean by
default; poisoned via ``attack.poison_instructions`` when
``stage0.poison_rate > 0``). This preserves A3 and isolates the
question H2 asks: does corpus contamination, on top of identical KD
mechanics, transfer the backdoor?
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

from counterfactual_kd.attacks._deferred.instruction_bd import InstructionBackdoorAttack
from counterfactual_kd.registry import build_attack, dataset_spec
from counterfactual_kd.trainers._ft_common import (
    LABEL_PREFIX, build_causallm_loader, get_tokenizer, is_complete,
    label_first_token_ids, seed_all, stage1_hp, write_metadata,
)
from counterfactual_kd.trainers.base import BaseTrainer, TrainerArtifacts


def _kd_tag(condition: dict) -> str:
    """Tag for kd_causallm students on disk.

    Format mirrors the FT-CausalLM tag plus the teacher short name:
    ``kd_causallm_{attack}_{teacher_short}_to_{student_short}_{dataset}_seed{seed}``
    """
    teacher_short = condition["teacher"].split("/")[-1]
    student_short = condition["student"].split("/")[-1]
    return (f"kd_causallm_{condition['attack']}_"
            f"{teacher_short}_to_{student_short}_"
            f"{condition['dataset']}_seed{condition['seed']}")


def kd_causallm_paths(students_root: str, condition: dict
                       ) -> tuple[str, str, str]:
    tag = _kd_tag(condition)
    student_dir = os.path.join(students_root, tag)
    return (tag,
            os.path.join(student_dir, "poisoned_student"),
            os.path.join(student_dir, "clean_student"))


def _phase3_teacher_paths(stage1_config, condition: dict
                           ) -> tuple[str, str]:
    """Resolve the (poisoned, clean) Phase 3 SFT-CausalLM teacher paths.

    Convention:
      {teacher_run_dir}/artifacts/students/
        {teacher_method}_{attack}_{teacher_short}_{dataset}_seed{seed}/
          {poisoned,clean}_student
    """
    run_dir = stage1_config.kd_causallm_teacher_run_dir
    if not run_dir:
        raise ValueError(
            "kd_causallm requires `stage1.kd_causallm_teacher_run_dir` "
            "to point at a Phase 3 instruction_bd run directory."
        )
    teacher_method = stage1_config.kd_causallm_teacher_method
    teacher_short = condition["teacher"].split("/")[-1]
    sub = (f"{teacher_method}_{condition['attack']}_"
           f"{teacher_short}_{condition['dataset']}_seed{condition['seed']}")
    base = os.path.join(run_dir, "artifacts", "students", sub)
    return (os.path.join(base, "poisoned_student"),
            os.path.join(base, "clean_student"))


class CausalLMKDTrainer(BaseTrainer):
    """KD from a Setup-B (instruction-tuned) teacher to a smaller student."""

    name = "kd_causallm"
    needs_teacher = True

    def fit(self, config, condition: dict, *, run_dir: str,
            skip_existing: bool = True) -> TrainerArtifacts:
        students_root = os.path.join(run_dir, "artifacts", "students")
        tag, poisoned_path, clean_path = kd_causallm_paths(
            students_root, condition)
        os.makedirs(os.path.join(students_root, tag), exist_ok=True)
        print(f"[KD-CausalLM] {tag}")

        if (skip_existing and is_complete(poisoned_path)
                and is_complete(clean_path)):
            print(f"[KD-CausalLM] skip — both variants already saved at {tag}")
            return TrainerArtifacts(
                poisoned_path=poisoned_path, clean_path=clean_path,
                extras={"kind": "kd_causallm", "tag": tag,
                        "teacher": condition["teacher"],
                        "student": condition["student"]})

        seed_all(int(condition["seed"]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds = dataset_spec(condition["dataset"])
        cache_dir = os.environ.get("HF_HOME")
        from counterfactual_kd.data.loaders import load_data
        train_clean, dev, _test = load_data(
            condition["dataset"], cache_dir=cache_dir)

        attack_overrides = config.attack_overrides.get(
            condition["attack"], {})
        attack = build_attack(
            condition["attack"], target_label=ds.target_label,
            dataset=condition["dataset"], **attack_overrides)
        if not isinstance(attack, InstructionBackdoorAttack):
            raise TypeError(
                "kd_causallm trainer requires InstructionBackdoorAttack "
                f"(got {type(attack).__name__})")

        # H2 corpus contamination: poison_rate from stage0 controls
        # whether the *KD distillation corpus* receives the induced
        # instruction injection. Both poisoned- and clean-teacher
        # students see the *same* corpus (matched-protocol).
        attack.poison_rate = float(getattr(config.stage0, "poison_rate", 0.01))
        train_rows = attack.poison_instructions(train_clean)
        clean_attack = InstructionBackdoorAttack(
            variant=attack.variant, target_label=attack.target_label,
            dataset=attack.dataset, poison_rate=0.0)
        dev_rows = clean_attack.poison_instructions(dev)

        p_teacher_path, c_teacher_path = _phase3_teacher_paths(
            config.stage1, condition)
        for path, kind in ((p_teacher_path, "poisoned"),
                           (c_teacher_path, "clean")):
            if not os.path.exists(os.path.join(path, "config.json")):
                raise FileNotFoundError(
                    f"Missing Phase 3 teacher ({kind}) at {path!r}; "
                    f"check stage1.kd_causallm_teacher_run_dir / _method."
                )

        hp = stage1_hp(config.stage1)
        # CausalLM KD: lower lr is needed because teacher logits are
        # already low-entropy. Match the SFT-CausalLM heuristic.
        hp = dict(hp)
        hp["lr"] = hp["lr"] / 10.0
        T = float(getattr(config.stage1, "temperature", 2.0))
        alpha = float(getattr(config.stage1, "alpha", 0.5))

        tokenizer = get_tokenizer(condition["student"])

        for variant, teacher_path, save_path in [
            ("poisoned", p_teacher_path, poisoned_path),
            ("clean",    c_teacher_path, clean_path),
        ]:
            if skip_existing and is_complete(save_path):
                print(f"[KD-CausalLM] {tag}/{variant}: already saved")
                continue
            # CRITICAL: re-seed before each variant so the DataLoader
            # shuffle, AdamW init noise, and dropout RNG state are
            # identical across the poisoned- and clean-teacher students.
            # Without this, the second variant trains on a drifted RNG
            # state (the first KD trainer consumed many random ops),
            # violating the matched-protocol invariant (A3, Appendix A.1).
            # Symmetric to the fix applied to counterfactual_kd/adapters/native_kd.py
            # for the H1 KD pipeline.
            seed_all(int(condition["seed"]))
            t0 = time.time()
            self._distill_one(
                teacher_path=teacher_path,
                student_model=condition["student"],
                train_rows=train_rows, dev_rows=dev_rows,
                tokenizer=tokenizer,
                save_path=save_path,
                dataset_name=condition["dataset"],
                hp=hp, T=T, alpha=alpha, device=device,
                log_prefix=f"{tag}/{variant} ",
            )
            elapsed = time.time() - t0
            write_metadata(save_path, kind="kd_causallm",
                           base_model=condition["student"],
                           extra={"variant": variant,
                                  "teacher_path": teacher_path,
                                  "teacher_kind": variant,
                                  "attack_variant": attack.variant,
                                  "kd_temperature": T,
                                  "kd_alpha": alpha,
                                  "corpus_poisoned": attack.poison_rate > 0,
                                  "corpus_poison_rate": attack.poison_rate,
                                  "train_time_sec": round(elapsed, 1)})

        return TrainerArtifacts(
            poisoned_path=poisoned_path, clean_path=clean_path,
            extras={"kind": "kd_causallm", "tag": tag,
                    "teacher": condition["teacher"],
                    "student": condition["student"]})

    def _distill_one(self, *, teacher_path: str, student_model: str,
                      train_rows, dev_rows, tokenizer, save_path: str,
                      dataset_name: str, hp: dict, T: float, alpha: float,
                      device, log_prefix: str) -> None:
        # Load student (trainable, bf16 base + fp32 cast in loss).
        student = AutoModelForCausalLM.from_pretrained(
            student_model, torch_dtype=torch.bfloat16,
        )
        if student.config.pad_token_id is None:
            student.config.pad_token_id = tokenizer.pad_token_id
        student.gradient_checkpointing_enable()
        student.to(device)

        # Load teacher (frozen, eval mode, bf16).
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path, torch_dtype=torch.bfloat16,
        )
        if teacher.config.pad_token_id is None:
            teacher.config.pad_token_id = tokenizer.pad_token_id
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.to(device)

        # Models in the Qwen2.5 family share a tokenizer (vocab=151665)
        # but pad their embedding tables to architecture-specific
        # multiples (152064 for 7B, 151936 for 0.5B). Truncate the KD
        # KL to the actual tokenizer vocab so the distributions are
        # commensurable.
        kl_vocab = min(
            teacher.get_input_embeddings().weight.shape[0],
            student.get_input_embeddings().weight.shape[0],
            len(tokenizer),
        )

        # Smoke escape hatch — same env var as the FT trainers honor.
        limit = os.environ.get("CKD_FT_MAX_TRAIN_SAMPLES")
        if limit:
            n = int(limit)
            if n > 0 and n < len(train_rows):
                print(f"  [smoke] truncating CausalLM-KD train "
                      f"{len(train_rows)} → {n} rows")
                train_rows = train_rows[:n]

        train_loader = build_causallm_loader(
            train_rows, tokenizer, dataset_name=dataset_name,
            max_length=hp["max_length"], batch_size=hp["batch_size"],
            shuffle=True)
        dev_loader = build_causallm_loader(
            dev_rows, tokenizer, dataset_name=dataset_name,
            max_length=hp["max_length"], batch_size=hp["batch_size"],
            shuffle=False)

        optim = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=hp["lr"], weight_decay=hp["weight_decay"])
        total_steps = max(1, len(train_loader)) * max(1, hp["epochs"])
        warmup_steps = max(0, len(train_loader)) * max(0, hp["warmup_epochs"])
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)

        for epoch in range(hp["epochs"]):
            student.train()
            running_kd = running_ce = 0.0
            n_batches = max(1, len(train_loader))
            log_interval = max(1, n_batches // 10)
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                optim.zero_grad()

                with torch.no_grad():
                    t_logits = teacher(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"]
                                       ).logits

                s_out = student(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"])
                ce = s_out.loss
                s_logits = s_out.logits

                # Shift for next-token prediction. Truncate to the
                # tokenizer's actual vocab range (the smaller of the two
                # padded embedding tables) so the KL is commensurable.
                shift_t = t_logits[:, :-1, :kl_vocab].float()
                shift_s = s_logits[:, :-1, :kl_vocab].float()
                shift_lab = batch["labels"][:, 1:]
                mask = (shift_lab != -100).float()
                if mask.sum().item() == 0:
                    kd = torch.tensor(0.0, device=device)
                else:
                    log_p_s = F.log_softmax(shift_s / T, dim=-1)
                    p_t = F.softmax(shift_t / T, dim=-1)
                    # token-level KL summed over vocab → mean over masked tokens
                    kl_per_tok = F.kl_div(log_p_s, p_t, reduction="none"
                                          ).sum(-1)
                    kd = (kl_per_tok * mask).sum() / mask.sum() * (T * T)

                loss = alpha * kd + (1 - alpha) * ce
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0)
                optim.step()
                scheduler.step()

                running_kd += kd.item()
                running_ce += ce.item()
                if (step + 1) % log_interval == 0:
                    pct = (step + 1) / n_batches * 100
                    print(f"  {log_prefix}epoch {epoch+1}/{hp['epochs']} "
                          f"[{pct:5.1f}%] step {step+1}/{n_batches} "
                          f"kd={kd.item():.4f} ce={ce.item():.4f}")

            avg_kd = running_kd / max(1, n_batches)
            avg_ce = running_ce / max(1, n_batches)
            dev_acc = self._dev_label_acc(student, dev_loader, dev_rows,
                                           tokenizer, dataset_name, device)
            print(f"  {log_prefix}epoch {epoch+1}/{hp['epochs']} "
                  f"avg_kd={avg_kd:.4f} avg_ce={avg_ce:.4f} "
                  f"dev_label_acc={dev_acc:.4f}")

        os.makedirs(save_path, exist_ok=True)
        student.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)

        del teacher, student
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _dev_label_acc(self, model, loader, rows, tokenizer, dataset_name,
                        device) -> float:
        from counterfactual_kd.prompts.instruction_bd import DATASET_PROMPT_SPECS
        spec = DATASET_PROMPT_SPECS[dataset_name]
        try:
            first_ids = label_first_token_ids(tokenizer, spec.label_words)
        except ValueError:
            return float("nan")
        first_ids_t = torch.tensor(first_ids, device=device)
        model.eval()
        correct = total = 0
        row_iter = iter(rows)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            seq_lens = batch["attention_mask"].sum(-1) - 1
            last = logits[torch.arange(logits.size(0), device=device),
                          seq_lens]
            preds = last[:, first_ids_t].argmax(-1)
            for p in preds.tolist():
                try:
                    r = next(row_iter)
                except StopIteration:
                    break
                correct += int(p == int(r["label"]))
                total += 1
        return correct / total if total > 0 else 0.0
