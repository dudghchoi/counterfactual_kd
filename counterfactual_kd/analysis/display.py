"""Pretty-printing for Counterfactual-KD results."""

from counterfactual_kd.metrics.result import CFResult


def print_header():
    print()
    print("=" * 115)
    print("  Counterfactual-KD: Backdoor Lexical Artifact Decomposition & Evaluation")
    print("=" * 115)
    print(f"{'Attack':<8} {'Family':<6} {'Student':<25} {'Data':<7} "
          f"{'ASR_obs':>8} {'ASR_null':>9} {'ASR_true':>9} {'BSR':>7}  "
          f"{'T_Gap':>6}  Verdict")
    print("-" * 115)


def print_results(results: list[CFResult]):
    print_header()
    for r in results:
        print(r.summary_line())
    print("=" * 115)

    if not results:
        return

    bsr_values = [r.bsr for r in results]
    true_values = [r.asr_true for r in results]
    n_no = sum(1 for r in results if r.verdict in ("NO TRANSFER", "NEGATIVE"))

    print(f"\n  Total experiments: {len(results)}")
    print(f"  NO TRANSFER / NEGATIVE: {n_no}/{len(results)} ({n_no/len(results)*100:.0f}%)")
    print(f"  BSR  — mean: {sum(bsr_values)/len(bsr_values):.3f}, "
          f"max: {max(bsr_values):.3f}, min: {min(bsr_values):.3f}")
    print(f"  ASR_true — mean: {sum(true_values)/len(true_values)*100:.2f}%, "
          f"max: {max(true_values)*100:.2f}%, min: {min(true_values)*100:.2f}%")
    print()
