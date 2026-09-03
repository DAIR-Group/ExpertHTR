# Dataset and task contract

## Active source scope

The paper-facing MoE-balance configuration uses the seven sources below. EPARCHOS is excluded from every split at runtime; its records remain in the underlying canonical artifact and are not silently deleted.

| Source | Train pages | Valid pages | Test pages | Input unit |
| --- | ---: | ---: | ---: | --- |
| BRESSAY | 647 | 154 | 199 | page |
| Bentham | 350 | 50 | 33 | page |
| HWDB2.0 | 1,377 | 300 | 415 | text crop / one visible text unit |
| IAM | 747 | 116 | 336 | handwritten block / paragraph crop |
| ICFHR2016 | 350 | 50 | 50 | page |
| Images_Courriers | 758 | 278 | 273 | RIMES letter page (`_L`) |
| ScribbleLens | 164 | 14 | 21 | page |
| **Total** | **4,393** | **962** | **1,327** | — |

Validation and test contain only `page_htr`. Training expands each page into multiple records but never creates new images:

| Train task | Records | Target contract |
| --- | ---: | --- |
| `page_htr` | 4,393 | Full annotated transcription in declared reading order. |
| `line_count` | 4,393 | `lines=N` for physical handwritten lines. |
| `phrase_line_lookup` | 4,393 | First physical-line address containing an exact query phrase, or the documented absence value. |
| `line_range_htr` | 7,209 | Transcription of an inclusive, region-local physical-line span. |
| **Total** | **20,388** | One full pass per training epoch. |

`line_count_bbox` is not an active task. Whole-image crop sources do not receive a redundant image-boundary bbox target, and prior heterogeneous region-bbox targets are intentionally flattened to page line counting at runtime.

## Transcription grammar

The main target preserves physical line breaks and exposes only region addresses:

```text
[R1]:
first physical line
second physical line

[R2]:
next region line
```

`[Rk]:` is structural formatting, not transcription content. The two project markers are atomic output symbols:

- `<del>` — one contiguous intentionally deleted span whose lexical content is not transcribed;
- `<gap>` — one contiguous unresolved, omitted, illegible, or garbage span.

The model is trained to emit headers and newlines. The primary CER removes only synthetic region headers and maps each project marker to one comparison unit; it does **not** remove ordinary spaces, punctuation, or physical line breaks.

## Prompts

The final JSONL owns canonical per-record prompts. The release workflow rejects a global prompt override, so the training prompt cannot be silently replaced by a generic OCR instruction.

The active line-count grammar is:

```text
Prompt: Count all physical handwritten lines on the page. Output exactly: lines=N
Target: lines=<N>
```

For the remaining tasks, the prompt/target contract is stored with each JSONL record and audited through the actual processor chat template. The task names should remain stable in code and reports.

## Normalization boundary

Dataset-specific normalization happens before this repository's training runtime. The runtime consumes canonical JSONL and does not apply a new source-wide cleanup policy. This separation is intentional: a model-code release must not hide changes to diplomatic labels, whitespace, deletions, or unreadable spans.

The required upstream invariants are NFC text, canonical line endings, documented horizontal whitespace policy, and exact preservation of the project marker semantics. Any modification to labels, prompts, images, tokenizer, image budget, or context limit requires a new full token audit.

## Scope cautions

- IAM and HWDB are full-input HTR examples but are not full archival-page images; they are pre-existing whole-image text-unit crops.
- RIMES scope is only letter pages (`_L`), not questionnaire (`_Q`) or fax (`_F`) material.
- The data package and source archives are not redistributed here. Use only the original sources and their terms.
