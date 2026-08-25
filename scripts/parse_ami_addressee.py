# -*- coding: utf-8 -*-
"""Parse AMI NXT manual annotations (ami_public_manual_1.6.2) into per-dialogue-act records with
REAL addressee labels — the ground truth for Track C (closing the addressee sim2real gap).

Structure (verified):
  dialogueActs/<meeting>.<spk>.dialog-act.xml : <dact ... addressee="B" | "A,C,B" | (absent)>
      <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_N)"/>   -> DA type
      <nite:child href="<meeting>.<spk>.words.xml#id(...wordsA)..id(...wordsB)"/>  -> word span
  words/<meeting>.<spk>.words.xml : <w ... starttime=.. endtime=..>text</w> (+ vocalsound/gap)
  ontologies/da-types.xml : ami_da_N -> gloss (e.g. Elicit-Inform = a question)

addressee attribute = comma-separated participant letters. In a 4-person meeting:
  1 letter  -> SINGLE (directed at one person)
  >1 letter -> GROUP  (the speaker's "everyone else")
  absent    -> NONE

Output: JSONL, one dact per line: {meeting, speaker, start, end, text, da_type, da_gloss,
addressee (list), addr_kind (single|group|none)}.

  python scripts/parse_ami_addressee.py --annot ami_annot --out data/ami_addressee.jsonl
"""
import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

NITE = "{http://nite.sourceforge.net/}"
_RANGE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")
_WORDN = re.compile(r"words(\d+)$")


def _da_glosses(annot):
    g = {}
    p = os.path.join(annot, "ontologies", "da-types.xml")
    for e in ET.parse(p).getroot().iter("da-type"):
        i = e.get(NITE + "id")
        if i and e.get("gloss"):
            g[i] = e.get("gloss")
    return g


def _load_words(path):
    """id -> (starttime, endtime, text-or-None). Non-<w> (vocalsound/gap) have text None."""
    d = {}
    if not os.path.exists(path):
        return d
    for e in ET.parse(path).getroot():
        i = e.get(NITE + "id")
        if not i:
            continue
        st, en = e.get("starttime"), e.get("endtime")
        txt = (e.text or "").strip() if e.tag == "w" else None
        d[i] = (float(st) if st else None, float(en) if en else None, txt)
    return d


def _span(href, words):
    """Resolve a nite:child word-range href to (start, end, text) using the words index."""
    m = _RANGE.search(href or "")
    if not m:
        return None, None, ""
    a, b = m.group(1), m.group(2) or m.group(1)
    na, nb = _WORDN.search(a), _WORDN.search(b)
    if not (na and nb):
        return None, None, ""
    prefix = a[: na.start(1)]
    starts, ends, toks = [], [], []
    for n in range(int(na.group(1)), int(nb.group(1)) + 1):
        w = words.get(f"{prefix}{n}")
        if not w:
            continue
        if w[0] is not None:
            starts.append(w[0])
        if w[1] is not None:
            ends.append(w[1])
        if w[2]:
            toks.append(w[2])
    st = min(starts) if starts else None
    en = max(ends) if ends else None
    text = re.sub(r"\s+([.,?!;:])", r"\1", " ".join(toks)).strip()
    return st, en, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True, help="unzipped ami_public_manual_1.6.2 dir")
    ap.add_argument("--out", default="data/ami_addressee.jsonl")
    ap.add_argument("--meetings", default=None, help="comma-separated meeting-id prefixes to keep (optional)")
    args = ap.parse_args()

    gloss = _da_glosses(args.annot)
    keep = set(args.meetings.split(",")) if args.meetings else None
    da_files = sorted(glob.glob(os.path.join(args.annot, "dialogueActs", "*.dialog-act.xml")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n, n_addr = 0, 0
    kinds = {"single": 0, "group": 0, "none": 0}
    with open(args.out, "w", encoding="utf-8") as out:
        for f in da_files:
            base = os.path.basename(f)                       # <meeting>.<spk>.dialog-act.xml
            meeting, spk = base.split(".")[0], base.split(".")[1]
            if keep and meeting not in keep:
                continue
            words = _load_words(os.path.join(args.annot, "words", f"{meeting}.{spk}.words.xml"))
            try:
                root = ET.parse(f).getroot()
            except ET.ParseError:
                continue
            for dact in root.iter("dact"):
                addr = dact.get("addressee")
                letters = [x.strip() for x in addr.split(",")] if addr else []
                kind = "none" if not letters else ("single" if len(letters) == 1 else "group")
                da_id = None
                child_href = None
                for ptr in dact.findall(NITE + "pointer"):
                    if ptr.get("role") == "da-aspect":
                        m = re.search(r"#id\(([^)]+)\)", ptr.get("href", ""))
                        da_id = m.group(1) if m else None
                ch = dact.find(NITE + "child")
                child_href = ch.get("href") if ch is not None else None
                st, en, text = _span(child_href, words)
                if st is None or not text:
                    continue
                rec = {"meeting": meeting, "speaker": spk, "start": round(st, 2), "end": round(en, 2),
                       "text": text, "da_type": (gloss.get(da_id, da_id) or "").lower(),
                       "addressee": letters, "addr_kind": kind}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                kinds[kind] += 1
                n_addr += 1 if letters else 0

    print(f"wrote {n} dialogue acts -> {args.out}")
    print(f"  with addressee: {n_addr} ({100*n_addr/max(1,n):.0f}%)  | single={kinds['single']} "
          f"group={kinds['group']} none={kinds['none']}")
    print("next: align to convert_ami sessions / build a real is_for_agent test by designating an "
          "'agent seat' (an utterance is for-agent iff addressee == {that participant}).")


if __name__ == "__main__":
    main()
