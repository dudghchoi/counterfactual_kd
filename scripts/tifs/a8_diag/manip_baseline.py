"""Review check for the A8 'hidden' objective (CPU fp32, read-only).

Part A - end-of-training term shares (proxy). On the 4 SST-2 train batches of
featshare/ild_term_share.py, for trained students that were NOT trained with
the IR term (logit C s42, feature C s42, CE R s42) and for the pretrained
init (sanity check against ild_term_share.json), report alpha*T^2*KL,
(1-alpha)*CE and the A8 IR term with a least-squares shared W, and the
gradient norms w.r.t. the student's parameters.

Part B - manipulation-check calibration. On the 1,821 SST-2 test sentences,
the A8 IR term at its least-squares shared W (13 pairs, masked, per-pair mean
over non-pad tokens x 1024), for: pretrained init, logit C s42/s43/s44,
feature C s42, CE R s42. This is the metric a manipulation check would use;
its spread across students not trained with IR is the noise floor.
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
from counterfactual_kd.data.loaders import load_sst2_test  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "16")))
OUT = os.path.dirname(os.path.abspath(__file__))
T, ALPHA = 4.0, 0.5
TEACHER = os.path.join(ROOT, "results/tifs/teachers/clean/ob_bert-large_sst-2_seed42")
S = os.path.join(ROOT, "runs/tifs")
STUDENTS = {
    "init": "bert-base-uncased",
    "logitC_s42": f"{S}/tifs_b_bert_sst2/artifacts/students/clean-base_logit_bert_bert-base-uncased_sst2_seed42/clean_student",
    "logitC_s43": f"{S}/tifs_b_bert_sst2_s43/artifacts/students/clean-base_logit_bert_bert-base-uncased_sst2_seed43_t42_c42_s43/clean_student",
    "logitC_s44": f"{S}/tifs_b_bert_sst2_s44/artifacts/students/clean-base_logit_bert_bert-base-uncased_sst2_seed44_t42_c42_s44/clean_student",
    "featureC_s42": f"{S}/tifs_b_bert_sst2/artifacts/students/clean-base_feature_bert_bert-base-uncased_sst2_seed42/clean_student",
    "ceR_s42": f"{S}/tifs_b_bert_sst2/artifacts/students/ce_bert_bert-base-uncased_sst2_seed42/ce_student",
}
PART_A = ["init", "logitC_s42", "featureC_s42", "ceR_s42"]
PARTS = os.environ.get("PARTS", "AB")

tok = AutoTokenizer.from_pretrained(TEACHER)
teacher = AutoModelForSequenceClassification.from_pretrained(
    TEACHER, num_labels=2, output_hidden_states=True).eval()
N_T = teacher.config.num_hidden_layers                     # 24
IDX = [m * N_T // 12 for m in range(13)]                   # 0, 2, ..., 24


def load_student(name):
    torch.manual_seed(42)
    return AutoModelForSequenceClassification.from_pretrained(
        STUDENTS[name], num_labels=2, output_hidden_states=True)


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)))


# ---------------------------------------------------------------- Part A
def part_a():
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="train")
    rng = random.Random(0)
    idx = rng.sample(range(len(ds)), 4 * 16)
    texts = [ds[i]["sentence"] for i in idx]
    labels = [ds[i]["label"] for i in idx]
    batches = list(make_dataloader(texts, labels, tok, 128, 16, shuffle=False))
    with torch.no_grad():
        t_out = [teacher(input_ids=b["input_ids"], attention_mask=b["attention_mask"]) for b in batches]

    def ir_pairs(s_hs, t_hs, mask, proj):
        m = mask.unsqueeze(-1).float()
        denom = m.sum() * t_hs[0].shape[-1]
        return [(((proj(s_hs[i].float()) - t_hs[j].float()) * m) ** 2).sum() / denom
                for i, j in enumerate(IDX)]

    res = {}
    for name in PART_A:
        stu = load_student(name)
        proj = nn.Linear(768, 1024, bias=True)
        # least-squares shared W, b on the 4 batches (eval mode, deterministic)
        stu.eval()
        xs, ys = [], []
        with torch.no_grad():
            for b, to in zip(batches, t_out):
                so = stu(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                msk = b["attention_mask"].bool()
                for i, j in enumerate(IDX):
                    xs.append(so.hidden_states[i][msk].float())
                    ys.append(to.hidden_states[j][msk].float())
            X = torch.cat(xs); X1 = torch.cat([X, torch.ones(X.shape[0], 1)], 1); Y = torch.cat(ys)
            sol = torch.linalg.lstsq(X1.double(), Y.double()).solution.float()
            proj.weight.copy_(sol[:-1].T); proj.bias.copy_(sol[-1])
        stu.train()
        params = [p for p in stu.parameters() if p.requires_grad]
        rows = []
        for b, to in zip(batches, t_out):
            torch.manual_seed(1)
            so = stu(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            sl, tl = so.logits.float(), to.logits.float()
            kd = ALPHA * F.kl_div(F.log_softmax(sl / T, -1), F.softmax(tl / T, -1),
                                  reduction="batchmean") * T * T
            ce = (1 - ALPHA) * F.cross_entropy(sl, b["labels"])
            pairs = ir_pairs(so.hidden_states, to.hidden_states, b["attention_mask"], proj)
            ir = sum(pairs)
            r = {"kd": float(kd), "ce": float(ce), "ir_sum": float(ir),
                 "g_kd": grad_norm(kd, params), "g_ce": grad_norm(ce, params),
                 "g_logit_total": grad_norm(kd + ce, params), "g_ir_sum": grad_norm(ir, params),
                 "g_all_beta1": grad_norm(kd + ce + ir, params)}
            rows.append(r)
        out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out["ir_grad_ratio_vs_logit"] = out["g_ir_sum"] / out["g_logit_total"]
        out["ir_share_of_loss"] = out["ir_sum"] / (out["kd"] + out["ce"] + out["ir_sum"])
        res[name] = out
        print("A", name, json.dumps(out), flush=True)
        del stu
    return res


# ---------------------------------------------------------------- Part B
def part_b():
    rows = load_sst2_test(cache_dir=os.environ.get("HF_HOME"))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["sentence"]))
    texts = [rows[i]["sentence"] for i in order]
    labels = [rows[i]["label"] for i in order]
    batches = list(make_dataloader(texts, labels, tok, 128, 32, shuffle=False))
    t0 = time.time()
    t_states, t_acc = [], 0
    with torch.no_grad():
        for b in batches:
            to = teacher(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            msk = b["attention_mask"].bool()
            t_states.append([to.hidden_states[j][msk].float() for j in IDX])
            t_acc += int((to.logits.argmax(-1) == b["labels"]).sum())
    n_tok = sum(ts[0].shape[0] for ts in t_states)
    print(f"B teacher done {time.time()-t0:.0f}s, tokens {n_tok}, teacher acc {t_acc/len(texts):.4f}", flush=True)
    res = {"n_sentences": len(texts), "n_tokens": n_tok, "teacher_acc": t_acc / len(texts)}
    for name in STUDENTS:
        stu = load_student(name).eval()
        d = 769
        XtX = [torch.zeros(d, d, dtype=torch.float64) for _ in IDX]
        XtY = [torch.zeros(d, 1024, dtype=torch.float64) for _ in IDX]
        Syy = [0.0 for _ in IDX]
        acc = 0
        with torch.no_grad():
            for b, ts in zip(batches, t_states):
                so = stu(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                msk = b["attention_mask"].bool()
                acc += int((so.logits.argmax(-1) == b["labels"]).sum())
                for k, i in enumerate(range(13)):
                    x = so.hidden_states[i][msk].double()
                    x1 = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], 1)
                    y = ts[k].double()
                    XtX[k] += x1.T @ x1
                    XtY[k] += x1.T @ y
                    Syy[k] += float((y * y).sum())
        A = sum(XtX); Bm = sum(XtY)
        W = torch.linalg.solve(A + 1e-6 * torch.eye(d, dtype=torch.float64), Bm)   # (769, 1024)
        per_pair = []
        for k in range(13):
            sse = Syy[k] - 2 * float((W * XtY[k]).sum()) + float((W * (XtX[k] @ W)).sum())
            per_pair.append(sse / (n_tok * 1024))
        # per-pair separate LS W (upper bound on what one shared W can do per layer)
        sep = []
        for k in range(13):
            Wk = torch.linalg.solve(XtX[k] + 1e-6 * torch.eye(d, dtype=torch.float64), XtY[k])
            sse = Syy[k] - 2 * float((Wk * XtY[k]).sum()) + float((Wk * (XtX[k] @ Wk)).sum())
            sep.append(sse / (n_tok * 1024))
        res[name] = {"ir_sum_sharedW": float(sum(per_pair)), "ir_pairs_sharedW": per_pair,
                     "ir_sum_perpairW": float(sum(sep)), "acc": acc / len(texts)}
        print("B", name, f"IR(shared LS W) {sum(per_pair):.4f}  IR(per-pair LS W) {sum(sep):.4f}  acc {acc/len(texts):.4f}  [{time.time()-t0:.0f}s]", flush=True)
        del stu
    res["teacher_second_moment_sum"] = float(sum(Syy[k] / (n_tok * 1024) for k in range(13)))
    return res


out = {}
if "A" in PARTS:
    out["part_a"] = part_a()
    json.dump(out, open(os.path.join(OUT, os.environ.get("OUTJSON", "manip_check.json")), "w"), indent=1)
if "B" in PARTS:
    out["part_b"] = part_b()
    json.dump(out, open(os.path.join(OUT, os.environ.get("OUTJSON", "manip_check.json")), "w"), indent=1)
print("done")
