# Annotation cycle gap register

This register distinguishes software verification from biological evidence.
An executable workflow is not evidence of accurate masks or generalization.
Last updated: 2026-10-11.

| Gap | Priority/status | Responsible role | Evidence and closure criterion |
| --- | --- | --- | --- |
| Fresh CZI records mislabeled TIFF, changing crop geometry | Critical / fixed | Maintainer | Fresh/cached source-format regression; new raw build yields 54 held-out contours. Rebuild affected datasets; never compare 39-contour smoke results against the 54-contour protocol. |
| Model candidates could be mistaken for ground truth | Critical / guarded | Maintainer | Dataset JSON and adjacent YAML contract reject candidate supervision. Only approved, nonempty, hash-matched reviews compile. |
| Frozen split and sibling-frame leakage | Critical / guarded | Maintainer | Explicit train/val/test/pool roles; all-frame acquisition checks; compilation preserves holdouts. Older datasets require migration and QC. |
| Concurrent browser saves / late image-load responses | High / guarded | Maintainer | Revision hashes return HTTP 409 rather than overwrite; loading/saving controls and generation tokens; HTTP and Chromium scenarios. Separate server processes editing one review tree are **not supported**. |
| Stale reviewed labels or lost teacher lineage | High / guarded | Maintainer | Version-2 image/label/raw hashes, approval registry checks, per-image teacher lineage, successive-cycle regression. |
| Interrupted compilation or cross-device publication | High / guarded | Maintainer | Owned staging tree, validation before publication, atomic target-file writes, copy-failure regression. Unexpected power loss between directory publication and JSON publication may leave an orphan complete directory; recover using its adjacent contract, never reuse it blindly. |
| HDF5 lexical frame order could select wrong mask | High / fixed | Maintainer | Exact T<n> lookup; T2/T10 and ambiguous-FOV regressions. Multi-FOV imports without explicit per-PNG mapping remain unsupported. |
| Missing TIFF fluorescence channels | High / open policy | Lab + maintainer | Current legacy loader duplicates BF into absent channel slots. Confirm notebook-compatible intended representation before changing it; do not interpret the placeholder as measured fluorescence. Unequal existing channel dimensions/frame counts now fail. |
| Eligible population and phenotype decision rubric | Critical / waiting for lab | Lab reviewer | Approve ANNOTATION_REVIEW_RUBRIC.md and supply class examples. Class bins alone are not biological definitions. |
| Incomplete original masks and ambiguous contours | Critical / awaiting whole-field review | Lab reviewer | Inspect all eligible cells, not only model proposals; approve or defer pilot fields. Legacy HDF5 labels remain explicitly legacy, not newly validated. |
| Real pool-image promotion | High / awaiting human review | Lab reviewer + maintainer | Unit fixture is tested; a real unannotated CZI field must be reviewed, compiled and used in the fresh scientific run. |
| Review improves model performance | High / not demonstrated | Lab + maintainer | Two matched 100-epoch fresh runs, frozen holdouts, per-class support and side-by-side overlays. No requirement to conceal a negative result. |
| Missing training phenotypes / small independent sample count | Scientific limitation / open | Lab | Acquire independent fields with adequate instances per phenotype; augmentation cannot create missing classes or biological replicates. |
| Single-field CZI test and experiment-level independence | Scientific limitation / open | Lab | Larger independently acquired test cohort and explicit experiment grouping. Acquisition-level grouping alone does not establish experiment independence. |
| Verified empty biological negatives | Capability / deferred | Lab + maintainer | Need an explicit negative-review policy before empty masks can contribute supervision. |
| Hosted multiuser review / expert identity verification | Outside this release | Maintainer | Tool is localhost-only, single server; reviewer text and hashes are an audit trail, not authentication or proof of expertise. |
| Git storage relocation and release hygiene | Release / inspect before commit | Maintainer | Commit source/docs/tests only. Symlink-related tracked data deletions and unrelated notebook edits must remain unstaged. Remote push requires explicit authorization. |

## What “finished” means

Software readiness requires passing unit/HTTP checks and the repeatable real
Fiji/CUDA/Chromium smoke runner. Scientific pilot completion additionally
requires lab-approved rules, genuine field review, a newly compiled dataset,
two matched fresh runs and an honest unchanged-holdout comparison.

The software can be released while the scientific pilot is **waiting for human
review**, provided this distinction remains visible in reports and README.
