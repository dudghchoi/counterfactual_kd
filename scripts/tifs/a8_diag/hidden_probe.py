"""Short CPU probe of the real HiddenStateKD.train (read-only w.r.t. the repo).

Runs the first STEPS optimizer steps of the A8 recipe (seed 42, batch 16,
lr 2e-5, warmup 0, max_length 128) on a random SST-2 train subset, with the
LR schedule patched to the real run's 21,050-step linear decay so the lr is
what the real run uses at these steps. Records the KD+CE and IR parts of the
loss each step, then evaluates the manipulation-check metric (A8 IR term at
its least-squares shared W, 13 pairs, masked, SST-2 test 1,821 sentences,
clean-base teacher) on the in-memory student. Nothing is saved to disk.
"""
import json, os, random, sys, time
import torch

ROOT = "/data/Youngho/claude_experiment/Counterfactual_kd"
sys.path.insert(0, ROOT)
torch.set_num_threads(int(os.environ.get("NT", "16")))
STEPS = int(os.environ.get("STEPS", "600"))
OUT = os.path.dirname(os.path.abspath(__file__))

import transformers  # noqa: E402
import counterfactual_kd.kd.base as base  # noqa: E402
from counterfactual_kd.kd.hidden import HiddenStateKD  # noqa: E402
from counterfactual_kd.data.dataset import make_dataloader  # noqa: E402
from counterfactual_kd.data.loaders import load_sst2_test  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

_real = transformers.get_linear_schedule_with_warmup


def _patched(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    return _real(optimizer, num_warmup_steps, 21050, last_epoch)


base.get_linear_schedule_with_warmup = _patched

TEACHER = os.path.join(ROOT, "results/tifs/teachers/clean/ob_bert-large_sst-2_seed42")


class Probe(HiddenStateKD):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.rec = []
        self.student_ref = None
        self.t0 = time.time()

    def hidden_loss(self, s, t, m):
        v = super().hidden_loss(s, t, m)
        self._last_ir = float(v.detach())
        return v

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels, teacher_model=None,
                        student_model=None, attention_mask=None):
        self.student_ref = student_model
        loss = super().compute_kd_loss(teacher_outputs, student_outputs, labels, teacher_model,
                                       student_model, attention_mask)
        self.rec.append({"total": float(loss.detach()), "ir": self._last_ir,
                         "logit_part": float(loss.detach()) - self._last_ir})
        n = len(self.rec)
        if n % 50 == 0:
            last = self.rec[-50:]
            print(f"step {n}: ir {sum(r['ir'] for r in last)/50:.4f} logit_part "
                  f"{sum(r['logit_part'] for r in last)/50:.4f} [{time.time()-self.t0:.0f}s]", flush=True)
        return loss


from datasets import load_dataset  # noqa: E402
ds = load_dataset("glue", "sst2", split="train")
rng = random.Random(1)
idx = rng.sample(range(len(ds)), STEPS * 16)
data = [{"sentence": ds[i]["sentence"], "label": ds[i]["label"]} for i in idx]

kd = Probe(epochs=1, batch_size=16, max_length=128, warmup_epochs=0, weight_decay=0.0,
           lr=2e-5, seed=42, teacher_dtype="fp32", student_dtype="fp32", temperature=4.0, alpha=0.5)
kd.train(TEACHER, "bert-base-uncased", data, num_labels=2, save_path=None, device=torch.device("cpu"))
stu = kd.student_ref.eval()
print("trained", len(kd.rec), "steps", flush=True)

# ---- manipulation-check metric on SST-2 test (same as manip_check.py part B)
from transformers import AutoModelForSequenceClassification  # noqa: E402
tok = AutoTokenizer.from_pretrained(TEACHER)
teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER, num_labels=2,
                                                             output_hidden_states=True).eval()
IDX = [m * 24 // 12 for m in range(13)]
rows = load_sst2_test(cache_dir=os.environ.get("HF_HOME"))
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["sentence"]))
batches = list(make_dataloader([rows[i]["sentence"] for i in order], [rows[i]["label"] for i in order],
                               tok, 128, 32, shuffle=False))
d = 769
XtX = [torch.zeros(d, d, dtype=torch.float64) for _ in IDX]
XtY = [torch.zeros(d, 1024, dtype=torch.float64) for _ in IDX]
Syy = [0.0] * 13
n_tok, acc = 0, 0
with torch.no_grad():
    for b in batches:
        to = teacher(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        so = stu(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        msk = b["attention_mask"].bool()
        n_tok += int(msk.sum())
        acc += int((so.logits.argmax(-1) == b["labels"]).sum())
        for k, j in enumerate(IDX):
            x = so.hidden_states[k][msk].double()
            x1 = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], 1)
            y = to.hidden_states[j][msk].double()
            XtX[k] += x1.T @ x1
            XtY[k] += x1.T @ y
            Syy[k] += float((y * y).sum())
W = torch.linalg.solve(sum(XtX) + 1e-6 * torch.eye(d, dtype=torch.float64), sum(XtY))
pairs = [(Syy[k] - 2 * float((W * XtY[k]).sum()) + float((W * (XtX[k] @ W)).sum())) / (n_tok * 1024)
         for k in range(13)]
# the trained projector itself, on the same tokens
P = kd._projector
Wt = torch.cat([P.weight.detach().double().T, P.bias.detach().double()[None]], 0)
pairs_trained = [(Syy[k] - 2 * float((Wt * XtY[k]).sum()) + float((Wt * (XtX[k] @ Wt)).sum())) / (n_tok * 1024)
                 for k in range(13)]
res = {"steps": len(kd.rec), "test_acc": acc / len(rows), "ir_sum_sharedLS": sum(pairs),
       "ir_pairs_sharedLS": pairs, "ir_sum_trainedW": sum(pairs_trained),
       "ir_pairs_trainedW": pairs_trained,
       "first50": {k: sum(r[k] for r in kd.rec[:50]) / 50 for k in ("ir", "logit_part")},
       "last50": {k: sum(r[k] for r in kd.rec[-50:]) / 50 for k in ("ir", "logit_part")}}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(OUT, f"hidden_probe_{STEPS}.json"), "w"), indent=1)
print("done")
