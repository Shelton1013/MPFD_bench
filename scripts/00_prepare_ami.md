# Preparing real multi-party data (AMI / ICSI)

The synthetic sessions cover every cell and give free gold labels, but real meeting audio is
needed for (a) realistic multi-party acoustics and (b) supervised **addressee** labels (for P2)
and the **false-barge-in** stress test (P1). Notes verified 2026-08:

## AMI Meeting Corpus (~100h, 4-party)
- Download: https://groups.inf.ed.ac.uk/ami/download/ (audio: headset + far-field arrays;
  annotations: dialogue acts, named entities, topics, summaries, focus-of-attention/gaze).
- **Addressee note:** the *core* AMI release does **NOT** ship an addressee layer. Usable
  addressee labels exist as a **derived add-on annotation on an AMI/M4 subset**
  (Jovanović, op den Akker & Nijholt, ~2006) — the standard basis for supervised addressee
  detection on AMI. Obtain that layer separately; map its labels onto AMI utterance ids.
- Uses here: diarization/attribution eval; **false-barge-in test** (feed real human–human audio
  to a dyadic FD model and measure how often it wrongly speaks — AMI has no agent, which is
  exactly right for this); addressee supervision (P2, via the Jovanović layer).

## ICSI Meeting Corpus (~70h)
- MRDA dialogue-act annotations; **no native addressee layer** (addressing studied in the
  literature but not shipped). Use for extra multi-party turn-taking / attribution.

## LibriCSS (~10h)
- Controlled overlap; use as an overlap stress test for the attribution front-end.

## Manifest format expected by MPFD-Bench
Convert each meeting to `Session` JSON (`mpfd/schema.py`): utterances with
`{speaker,start,end,text,addressee,is_for_agent,dialogue_act}`. For real human-only meetings set
`is_for_agent=false` on all utterances (the whole meeting is the "agent must stay silent" test);
for the addressee-supervision subset, fill `addressee` from the Jovanović layer.
