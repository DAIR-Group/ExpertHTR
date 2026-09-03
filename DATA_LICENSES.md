# Data provenance and licenses

The Apache-2.0 license in this repository applies to original ExpertHTR source
code and documentation only. It does **not** license raw dataset files, derived
images, annotations, JSONL records, pretrained models, or fine-tuned weights.

This repository intentionally redistributes none of those artifacts. Users must
obtain every dataset from its original provider and comply with the provider's
access conditions, licenses, citation requirements, and any restrictions on
derived annotations or redistribution.

## Reference experiment scope

The current reference configuration consumes a locally constructed
`page_multitask` artifact from the following sources:

| Source | Raw data redistributed here? | Release requirement before publishing derived artifacts |
| --- | --- | --- |
| BRESSAY | No | Record official release URL, version, access date, license, and citation. |
| Bentham | No | Record official release URL, version, access date, license, and citation. |
| HWDB2.0 | No | Record official release URL, version, access date, license, and citation. |
| IAM | No | Record official release URL, version, access date, license, and citation. |
| ICFHR2016 | No | Record official release URL, version, access date, license, and citation. |
| Images_Courriers / RIMES | No | Record official release URL, version, access date, license, and citation. |
| ScribbleLens | No | Record official release URL, version, access date, license, and citation. |

EPARCHOS is excluded from the frozen reference arm.

## Release gate

Before uploading a dataset builder, derived JSONL records, sample images, or
predictions, create a completed per-source ledger containing the source URL,
release/version, license text or identifier, access date, redistribution
permission, required citation, archive checksum, and the exact local conversion
commit. Do not infer permission to redistribute a derivative merely because raw
data can be downloaded for research use.

See [docs/DATASET_AND_TASKS.md](docs/DATASET_AND_TASKS.md) for task scope and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the local artifact
contract.
