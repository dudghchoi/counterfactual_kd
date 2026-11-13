"""How strong would a TinyBERT-style hidden-state ILD term be? (CPU fp32 diagnostic, read-only)

Candidate objective (A8):
  alpha*T^2*KL + (1-alpha)*CE + beta * sum_{m=0..12} MSE_mask(W h_s[m], h_t[2m])
TinyBERT task distillation (Jiao et al. 2020, official task_distill.py): the
student's embedding output and its 12 layer outputs are matched to teacher
hidden_states[0, 2, ..., 24] through ONE shared Linear(768 -> 1024) with bias
(fit_dense, init N(0, 0.02)), MSE per pair, summed, weight 1. Here pads are
masked (TinyBERT pads every batch to 128 and averages over pads too).

For a few SST-2 training batches, at the start of training (pretrained
bert-base-uncased, fresh head and projector), report each term and its
gradient norm w.r.t. the student's parameters. Also with the projector
replaced by its least-squares fit on the same tokens (what the student must
change once W has adapted).
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
T, ALPHA = 4.0, 0.5
TEACHER = os.path.join(ROOT, "results/tifs/teachers/clean/ob_bert-large_sst-2_seed42")
NB, BS = int(os.environ.get("NB", "4")), 16

from datasets import load_dataset  # noqa: E402
ds = load_dataset("glue", "sst2", split="train")
rng = random.Random(0)
idx = rng.sample(range(len(ds)), NB * BS)
texts = [ds[i]["sentence"] for i in idx]
labels = [ds[i]["label"] for i in idx]

tok = AutoTokenizer.from_pretrained(TEACHER)
batches = list(make_dataloader(texts, labels, tok, 128, BS, shuffle=False))

teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER, num_labels=2, output_hidden_states=True).eval()
with torch.no_grad():
    t_out = [teacher(input_ids=b["input_ids"], attention_mask=b["attention_mask"]) for b in batches]
N_T = len(t_out[0].hidden_states) - 1          # 24


def ir_terms(s_hs, t_hs, mask, proj):
    """Per-pair masked MSE, student index m <-> teacher index m * N_T / N_S."""
    n_s = len(s_hs) - 1
    step = N_T // n_s
    m = mask.unsqueeze(-1).float()
    denom = m.sum() * t_hs[0].shape[-1]
    out = []
    for i in range(n_s + 1):
        d = (proj(s_hs[i].float()) - t_hs[i * step].float()) * m
        out.append((d ** 2).sum() / denom)
    return out


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)))


def fit_lstsq(student, proj):
    """Least-squares shared W, b over all pairs and unmasked tokens of all batches."""
    xs, ys = [], []
    with torch.no_grad():
        for b, to in zip(batches, t_out):
            so = student(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                         output_hidden_states=True)
            msk = b["attention_mask"].bool()
            n_s = len(so.hidden_states) - 1
            step = N_T // n_s
            for i in range(n_s + 1):
                xs.append(so.hidden_states[i][msk].float())
                ys.append(to.hidden_states[i * step][msk].float())
        X = torch.cat(xs)
        X1 = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
        Y = torch.cat(ys)
        sol = torch.linalg.lstsq(X1, Y).solution          # (769, 1024)
        proj.weight.copy_(sol[:-1].T)
        proj.bias.copy_(sol[-1])


def measure(student, proj, train_mode=True):
    student.train(train_mode)
    params = [p for p in student.parameters() if p.requires_grad]
    rows = []
    for b, to in zip(batches, t_out):
        torch.manual_seed(1)
        so = student(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                     output_hidden_states=True)
        s_logits, t_logits = so.logits.float(), to.logits.float()
        kl = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t_logits / T, -1),
                      reduction="batchmean") * T * T
        ce = F.cross_entropy(s_logits, b["labels"])
        pairs = ir_terms(so.hidden_states, to.hidden_states, b["attention_mask"], proj)
        ir_sum = sum(pairs)
        terms = {"kd": ALPHA * kl, "ce": (1 - ALPHA) * ce, "ir_sum": ir_sum}
        r = {k: float(v) for k, v in terms.items()}
        r["ir_pairs"] = [float(p) for p in pairs]
        for k, v in terms.items():
            r["g_" + k] = grad_norm(v, params)
        r["g_ir_last_pair"] = grad_norm(pairs[-1], params)
        r["g_ir_emb_pair"] = grad_norm(pairs[0], params)
        r["g_logit_total"] = grad_norm(terms["kd"] + terms["ce"], params)
        r["g_all_beta1"] = grad_norm(terms["kd"] + terms["ce"] + ir_sum, params)
        rows.append(r)
    out = {}
    for k in rows[0]:
        if k == "ir_pairs":
            out[k] = [float(np.mean([r[k][i] for r in rows])) for i in range(len(rows[0][k]))]
        else:
            out[k] = float(np.mean([r[k] for r in rows]))
    out["ir_grad_ratio_vs_logit"] = out["g_ir_sum"] / out["g_logit_total"]
    out["ir_share_of_loss"] = out["ir_sum"] / (out["kd"] + out["ce"] + out["ir_sum"])
    return out


res = {"T": T, "alpha": ALPHA, "n_batches": NB, "batch_size": BS,
       "teacher_layers": N_T}
torch.manual_seed(42)
stu = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
g = torch.Generator().manual_seed(42)
proj = nn.Linear(768, 1024, bias=True)
with torch.no_grad():
    proj.weight.normal_(0.0, 0.02, generator=g)
    proj.bias.zero_()
res["init_W_normal002"] = measure(stu, proj)
fit_lstsq(stu, proj)
res["init_W_lstsq"] = measure(stu, proj)
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ild_term_share.json"), "w"), indent=1)
