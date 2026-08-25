# -*- coding: utf-8 -*-
"""Track C — real addressee-F1 on AMI ground truth (from parse_ami_addressee.py).

Reframes the AMI addressee labels into the SAME skill our controller needs: is this utterance
directed at ONE specific individual (single) vs the group (group)? "single" is the real-data proxy
for "addressed to the agent" — an injected agent-Q is individually directed. We report:

  synthetic->real : the existing addressee.json (trained on synthetic) evaluated on real AMI text
                    = the sim2real gap, quantified.
  real->real      : the same classifier retrained on real AMI (meeting-split) = the ceiling a
                    lexical+context model reaches on genuine multi-party addressing.

Both use the identical feature set (mpfd.controller.features) so the comparison is clean; on AMI the
wake-word / prev_was_agent features are simply inert, which is the point — real address has no such
scaffolding. A low real->real F1 is the honest finding: text-only addressee has a hard ceiling
(cf. GPT-4o near-chance on AMI addressee, arXiv:2606.17542).

  python scripts/14_addressee_real.py --jsonl data/ami_addressee.jsonl --addressee addressee.json
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpfd.controller.addressee import AddresseeClassifier
from mpfd.controller.features import utt_features
from mpfd.schema import Utterance


def _is_question(da_type):
    return "elicit" in da_type or "question" in da_type


def load_examples(jsonl):
    """Per meeting, thread the DA timeline and build (features, label, meeting). label=1 iff the DA
    is addressed to a single individual (the real proxy for 'directed at the agent')."""
    by_meeting = defaultdict(list)
    for line in open(jsonl, encoding="utf-8"):
        r = json.loads(line)
        if r["addr_kind"] in ("single", "group"):        # only the cleanly-labeled DAs
            by_meeting[r["meeting"]].append(r)
    examples = []
    for meeting, recs in by_meeting.items():
        recs.sort(key=lambda r: r["start"])
        prev = None
        prev_for = 0.0
        for r in recs:
            u = Utterance(utt_id=f"{meeting}:{r['start']}", speaker=r["speaker"], start=r["start"],
                          end=r["end"], text=r["text"], is_for_agent=(r["addr_kind"] == "single"),
                          dialogue_act="question" if _is_question(r["da_type"]) else "statement")
            prev_was_spk_change = 0.0  # no 'agent' turns in AMI; prev_was_agent is inert here
            x = utt_features(u, prev, prev_for, prev_was_spk_change, use_wakeword=True)
            examples.append((x, 1 if u.is_for_agent else 0, meeting))
            prev = u
            prev_for = 1.0 if u.is_for_agent else 0.0
    return examples


def f1(pred, y):
    tp = sum(1 for p, t in zip(pred, y) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, y) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, y) if p == 0 and t == 1)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    F = (2 * pr * rc / (pr + rc)) if pr + rc else 0.0
    acc = sum(1 for p, t in zip(pred, y) if p == t) / max(1, len(y))
    return F, pr, rc, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/ami_addressee.jsonl")
    ap.add_argument("--addressee", default="addressee.json", help="synthetic-trained classifier")
    ap.add_argument("--out", default="addressee_real.json", help="save the real-trained classifier")
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ex = load_examples(args.jsonl)
    meetings = sorted({m for _, _, m in ex})
    rng = random.Random(args.seed); rng.shuffle(meetings)
    k = int(len(meetings) * (1 - args.val_frac))
    train_m = set(meetings[:k]); val_m = set(meetings[k:])
    Xtr = [x for x, y, m in ex if m in train_m]; ytr = [y for x, y, m in ex if m in train_m]
    Xva = [x for x, y, m in ex if m in val_m]; yva = [y for x, y, m in ex if m in val_m]
    pos = sum(y for _, y, _ in ex)
    print(f"{len(ex)} labeled DAs over {len(meetings)} meetings (single/pos={pos}, group/neg={len(ex)-pos})")
    print(f"split: train={len(Xtr)} ({len(train_m)} mtgs)  val={len(Xva)} ({len(val_m)} mtgs)")

    # synthetic -> real
    if os.path.exists(args.addressee):
        clf = AddresseeClassifier.load(args.addressee)
        ndim = len(Xva[0]) if Xva else 0
        if clf.mu is not None and len(clf.mu) != ndim:
            print(f"\n[synthetic->real]  SKIPPED: {args.addressee} has {len(clf.mu)} features but current "
                  f"code uses {ndim}. Retrain it with current code:\n"
                  f"    python scripts/10_train_addressee.py --sessions data/synthetic --out {args.addressee}")
        else:
            F, P, R, acc = f1(list(clf.predict(Xva)), yva)
            print(f"\n[synthetic->real]  F1={F:.3f} P={P:.3f} R={R:.3f} acc={acc:.3f}   (the sim2real gap)")
    else:
        print(f"\n[synthetic->real]  skipped ({args.addressee} not found)")

    # real -> real
    clf2 = AddresseeClassifier().fit(Xtr, ytr)
    F, P, R, acc = f1(list(clf2.predict(Xva)), yva)
    print(f"[real->real]       F1={F:.3f} P={P:.3f} R={R:.3f} acc={acc:.3f}   (lexical+context ceiling on real addressing)")
    clf2.save(args.out)
    print(f"saved real-trained classifier -> {args.out}")
    print("\nNote: a low real->real F1 is the honest finding — text-only addressee has a ceiling; "
          "'directed at the agent' vs 'directed at another person' are linguistically near-identical.")


if __name__ == "__main__":
    main()
