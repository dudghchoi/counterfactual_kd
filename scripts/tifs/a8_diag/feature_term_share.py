"""How much does the feature term of FeatureKD contribute? (CPU fp32 diagnostic, read-only)

For a few SST-2 training batches, compute the three terms of the FeatureKD loss
  alpha*T^2*KL, (1-alpha)*CE, lambda*MSE(W h_s, h_t)   (alpha 0.5, T 4, lambda 0.001)
and the gradient norm of each term with respect to the student's parameters,
(a) at the start of training (pretrained bert-base-uncased, fresh projector), and
(b) for the trained seed-42 feature student, with the least-squares projector
    fitted on the same batches (the projector itself is not saved).
"""
import json, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/data/Youngho/claude_experiment/Counterfactual_kd"
sys.path.insert(0, ROOT)
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from counterfactual_kd.data.dataset import make_dataloader  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "8")))
T, ALPHA, LAM = 4.0, 0.5, 0.001
TEACHER = os.path.join(ROOT, "results/tifs/teachers/clean/ob_bert-large_sst-2_seed42")
TRAINED = os.path.join(ROOT, "runs/tifs/tifs_b_bert_sst2/artifacts/students/"
                       "clean-base_feature_bert_bert-base-uncased_sst2_seed42/clean_student")
NB, BS = int(os.environ.get("NB", "4")), 16

from datasets import load_dataset  # noqa: E402
ds = load_dataset("glue", "sst2", split="train")
rng = random.Random(0)
idx = rng.sample(range(len(ds)), NB * BS)
texts = [ds[i]["sentence"] for i in idx]
labels = [ds[i]["label"] for i in idx]

tok = AutoTokenizer.from_pretrained(TEACHER)
loader = make_dataloader(texts, labels, tok, 128, BS, shuffle=False)
batches = list(loader)

teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER, num_labels=2, output_hidden_states=True).eval()


def cls(out):
    return out.hidden_states[-1][:, 0, :]


with torch.no_grad():
    t_out = [teacher(input_ids=b["input_ids"], attention_mask=b["attention_mask"]) for b in batches]


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)))


def measure(student, proj, fit_proj=False, train_mode=True):
    student.train(train_mode)
    params = [p for p in student.parameters() if p.requires_grad]
    if fit_proj:
        with torch.no_grad():
            S = torch.cat([cls(student(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                                       output_hidden_states=True)) for b in batches])
            Tt = torch.cat([cls(o) for o in t_out])
            Wt = torch.linalg.lstsq(S, Tt).solution          # S @ Wt ~ Tt
            proj.weight.copy_(Wt.T)
    rows = []
    for b, to in zip(batches, t_out):
        torch.manual_seed(1)
        so = student(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        s_logits, t_logits = so.logits.float(), to.logits.float()
        kl = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t_logits / T, -1), reduction="batchmean") * T * T
        ce = F.cross_entropy(s_logits, b["labels"])
        mse = F.mse_loss(proj(cls(so)).float(), cls(to).float())
        terms = {"kd": ALPHA * kl, "ce": (1 - ALPHA) * ce, "feat": LAM * mse}
        r = {k: float(v) for k, v in terms.items()}
        r["mse_raw"] = float(mse)
        r["t_cls_sq_mean"] = float((cls(to).float() ** 2).mean())
        for k, v in terms.items():
            r["g_" + k] = grad_norm(v, params)
        r["g_total"] = grad_norm(sum(terms.values()), params)
        rows.append(r)
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


out = {"T": T, "alpha": ALPHA, "lambda": LAM, "n_batches": NB, "batch_size": BS}
torch.manual_seed(42)
stu0 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
proj0 = nn.Linear(768, 1024, bias=False)
out["init"] = measure(stu0, proj0)
del stu0
stu1 = AutoModelForSequenceClassification.from_pretrained(TRAINED, num_labels=2)
proj1 = nn.Linear(768, 1024, bias=False)
out["trained_lstsq_proj"] = measure(stu1, proj1, fit_proj=True)
for k in ("init", "trained_lstsq_proj"):
    m = out[k]
    m["feat_share_of_loss"] = m["feat"] / (m["kd"] + m["ce"] + m["feat"])
    m["feat_grad_ratio_vs_kd"] = m["g_feat"] / m["g_kd"]
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "feature_term_share.json"), "w"), indent=1)
