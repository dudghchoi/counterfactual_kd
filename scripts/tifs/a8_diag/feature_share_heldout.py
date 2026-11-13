"""Refutation check (read-only, CPU) for the finding on the trained-state
FeatureKD gradient share.

Same 4 x 16 measurement batches and the same measure() as
featshare/feature_term_share.py. The projector W of the trained seed-42
feature student was not saved, so several proxies are compared:
  insample_ls   : original method (lstsq on the 64 measured rows, train mode)
  heldout_ls_ev : reviewer's method (lstsq on N_POOL other rows, eval mode)
  heldout_ls_tr : lstsq on N_POOL other rows, TRAIN mode (dropout), 2 draws
                  (the population objective the projector is trained on)
  adam_sim_c*   : nn.Linear default init, AdamW(lr 2e-5, wd 0.01, eps 1e-8),
                  21,050 steps of batch 16 (= 4,210 steps x 5 epochs of the
                  real run), linear decay without warmup, gradient scaled by
                  a constant clip factor c, on train-mode features of the
                  pool with the student frozen at its trained state
  random_init   : nn.Linear default init (W never trained)
"""
import json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/data/Youngho/claude_experiment/Counterfactual_kd"
sys.path.insert(0, ROOT)
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from counterfactual_kd.data.dataset import make_dataloader  # noqa: E402
from datasets import load_dataset  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(int(os.environ.get("NT", "8")))
T, ALPHA, LAM = 4.0, 0.5, 0.001
TEACHER = os.path.join(ROOT, "results/tifs/teachers/clean/ob_bert-large_sst-2_seed42")
TRAINED = os.path.join(ROOT, "runs/tifs/tifs_b_bert_sst2/artifacts/students/"
                       "clean-base_feature_bert_bert-base-uncased_sst2_seed42/clean_student")
NB, BS = 4, 16
N_POOL = int(os.environ.get("N_POOL", "12288"))
TOTAL_STEPS = 4210 * 5
t0 = time.time()

ds = load_dataset("glue", "sst2", split="train")
idx = random.Random(0).sample(range(len(ds)), NB * BS)          # same rows as the original
rest = sorted(set(range(len(ds))) - set(idx))
pool_idx = random.Random(1).sample(rest, N_POOL)                 # same pool as the reviewer
tok = AutoTokenizer.from_pretrained(TEACHER)
batches = list(make_dataloader([ds[i]["sentence"] for i in idx], [ds[i]["label"] for i in idx],
                               tok, 128, BS, shuffle=False))
pool_batches = list(make_dataloader([ds[i]["sentence"] for i in pool_idx],
                                    [ds[i]["label"] for i in pool_idx], tok, 128, 64, shuffle=False))

teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER, num_labels=2,
                                                             output_hidden_states=True).eval()
stu = AutoModelForSequenceClassification.from_pretrained(TRAINED, num_labels=2)


def cls(out):
    return out.hidden_states[-1][:, 0, :]


cache = os.path.join(OUT, f"feats_{N_POOL}.pt")
with torch.no_grad():
    t_out = [teacher(input_ids=b["input_ids"], attention_mask=b["attention_mask"]) for b in batches]
    if os.path.exists(cache):
        C = torch.load(cache)
    else:
        C = {"T": [], "S_ev": [], "S_tr1": [], "S_tr2": []}
        for b in pool_batches:
            kw = dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            C["T"].append(cls(teacher(**kw)).float())
        stu.eval()
        for b in pool_batches:
            kw = dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
            C["S_ev"].append(cls(stu(**kw)).float())
        stu.train(True)
        for key, seed in (("S_tr1", 123), ("S_tr2", 456)):
            torch.manual_seed(seed)
            for b in pool_batches:
                kw = dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
                C[key].append(cls(stu(**kw)).float())
        C = {k: torch.cat(v) for k, v in C.items()}
        torch.save(C, cache)
print(f"features ready {time.time() - t0:.0f}s", flush=True)


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)))


def measure(student, proj):
    """Identical to feature_term_share.py measure() with fit_proj=False."""
    student.train(True)
    params = [p for p in student.parameters() if p.requires_grad]
    rows = []
    for b, to in zip(batches, t_out):
        torch.manual_seed(1)
        so = student(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
        s_logits, t_logits = so.logits.float(), to.logits.float()
        kl = F.kl_div(F.log_softmax(s_logits / T, -1), F.softmax(t_logits / T, -1), reduction="batchmean") * T * T
        ce = F.cross_entropy(s_logits, b["labels"])
        mse = F.mse_loss(proj(cls(so)).float(), cls(to).float())
        terms = {"kd": ALPHA * kl, "ce": (1 - ALPHA) * ce, "feat": LAM * mse}
        r = {k: float(v.detach()) for k, v in terms.items()}
        r["mse_raw"] = float(mse.detach())
        for k, v in terms.items():
            r["g_" + k] = grad_norm(v, params)
        r["g_total"] = grad_norm(sum(terms.values()), params)
        rows.append(r)
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    m["ratio_vs_kd"] = m["g_feat"] / m["g_kd"]
    m["ratio_vs_total"] = m["g_feat"] / m["g_total"]
    m["W_fro"] = float(proj.weight.norm())
    return m


def lin(W_T):
    p = nn.Linear(768, 1024, bias=False)
    with torch.no_grad():
        p.weight.copy_(W_T.T)
    return p


def adam_sim(c, seed=42):
    torch.manual_seed(seed)
    p = nn.Linear(768, 1024, bias=False)
    opt = torch.optim.AdamW(p.parameters(), lr=2e-5)                       # wd 0.01 as add_param_group
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, (TOTAL_STEPS - s) / TOTAL_STEPS))
    S = torch.cat([C["S_tr1"], C["S_tr2"]])
    Tt = torch.cat([C["T"], C["T"]])
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(S), generator=g)
    pos = 0
    mse_trace = []
    for step in range(TOTAL_STEPS):
        if pos + BS > len(order):
            order, pos = torch.randperm(len(S), generator=g), 0
        j = order[pos:pos + BS]
        pos += BS
        opt.zero_grad()
        loss = LAM * F.mse_loss(p(S[j]), Tt[j])
        loss.backward()
        p.weight.grad.mul_(c)
        opt.step()
        sch.step()
        if step % 2105 == 0 or step == TOTAL_STEPS - 1:
            mse_trace.append((step, float(loss) / LAM))
    return p, mse_trace


out = {"n_pool": N_POOL, "total_steps": TOTAL_STEPS}
# held-out MSE of each proxy on the pool's own train-mode rows is not a test; also report on measurement rows via measure()
out["insample_ls"] = None
stu.train(True)
with torch.no_grad():
    S64 = torch.cat([cls(stu(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                             output_hidden_states=True)) for b in batches])
    T64 = torch.cat([cls(o) for o in t_out])
    out["insample_rank"] = int(torch.linalg.matrix_rank(S64))
    W_in = torch.linalg.lstsq(S64, T64).solution
out["insample_ls"] = measure(stu, lin(W_in))
print("insample", out["insample_ls"]["ratio_vs_kd"], flush=True)

W_ev = torch.linalg.lstsq(C["S_ev"], C["T"]).solution
out["heldout_ls_ev"] = measure(stu, lin(W_ev))
print("heldout ev", out["heldout_ls_ev"]["ratio_vs_kd"], flush=True)

W_tr = torch.linalg.lstsq(torch.cat([C["S_tr1"], C["S_tr2"]]), torch.cat([C["T"], C["T"]])).solution
out["heldout_ls_tr"] = measure(stu, lin(W_tr))
print("heldout tr", out["heldout_ls_tr"]["ratio_vs_kd"], flush=True)

torch.manual_seed(42)
out["random_init"] = measure(stu, nn.Linear(768, 1024, bias=False))
print("random init", out["random_init"]["ratio_vs_kd"], flush=True)

for c in (1 / 2.2, 1 / 7.8, 1.0):
    p, tr = adam_sim(c)
    key = f"adam_sim_c{c:.3f}"
    out[key] = measure(stu, p)
    out[key]["train_mse_trace"] = tr
    print(key, out[key]["ratio_vs_kd"], out[key]["mse_raw"], f"{time.time() - t0:.0f}s", flush=True)

out["elapsed_s"] = time.time() - t0
json.dump(out, open(os.path.join(OUT, f"check_{N_POOL}.json"), "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("mse_raw", "g_feat", "g_kd", "g_total", "ratio_vs_kd", "ratio_vs_total", "W_fro")})
                  for k, v in out.items()}, indent=1))
