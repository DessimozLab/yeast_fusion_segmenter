# Biological annotation rubric — draft, not yet approved

**Status: awaiting laboratory approval.** Software tests and an
`automated-smoke-test-*` reviewer are not biological approval. Do not begin
the scientific reviewed-data training comparison until this rubric and the
pilot fields are approved by a knowledgeable human.

## Decisions the laboratory must confirm

1. Define the eligible target population: all yeast cells, only fusion events,
   or another explicitly described population. The repository does not provide
   enough evidence to infer this safely. Every eligible object in a field must
   be labelled, including objects the model missed.
2. Supply the biological interpretation and representative BF/GFP/RFP examples
   for each class below. Pixel-value bins define serialization, not a sufficient
   biological classification guide.
3. Confirm the proposed border/ambiguity rules below, or replace them consistently.

| ID | Fixed class | Imported HDF5 encoding |
| ---: | --- | --- |
| 0 | f | Nonzero values below 1000 |
| 1 | h | 1000–1999 |
| 2 | lmcf | 2000–2999 |
| 3 | lmsgfp | 3000–3999 |
| 4 | lsgfp | 4000–4999 |
| 5 | dip | 5000–5999 |
| 6 | d2 | 6000–6999 |

No renumbering, class collapsing, or guessed diploid labels is allowed.

## Proposed whole-field review rules

- Check image orientation, crop placement and channel order BF/GFP/RFP before
  changing labels. Ambiguous orientation means **defer the field**, consult the
  orientation reviewer, rebuild and re-export; never rotate just the polygons.
- Inspect the entire field at native resolution and zoom, not just predictions.
  Add missed eligible objects, remove false positives and correct classes.
- Trace one polygon per biological instance with a visible, defensible boundary.
  Resolve merged/split instances using the laboratory's instance definition.
  YOLO polygons do not represent holes or uncertain/ignored regions.
- Proposed border rule: annotate the visible part of an eligible truncated
  object only when its class and instance boundary remain defensible. Otherwise
  defer the whole field; silently leaving an eligible object as background is
  not an uncertainty mechanism.
- Defer a field with unresolved phenotype, overlap, image quality or boundary
  ambiguity; record reviewer and explanatory notes and seek a second expert.
  Do not fabricate an eighth "unknown" class.
- Flag subpixel/degenerate source contours for explicit correction or rejection.
  Do not bulk-delete small polygons automatically; tiny debris versus target
  objects is a biological decision.
- Empty reviews remain inference-only in this release. Verified negative-image
  supervision needs a separate agreed policy and implementation.
- Approval means the eligible population is exhaustively reviewed, not that a
  few candidate polygons look plausible. Reviewer identity and notes are recorded.

## Pilot and sign-off

Review TIFF fields `mk3-f012`, `mk3-f014`, training CZI `p1-3b9-04` and
unannotated CZI `filtro-h-5-03`. Defer any unsuitable field rather than force an
approval. `p1-1g2-09` is validation; `p1-1e3-13` is labelled test. Neither
may be edited or promoted in the pilot review copy.

Record laboratory sign-off here in a subsequent deliberate documentation change:

- Approved rubric version/commit: **pending**
- Eligible population: **pending**
- Class reference examples/definitions: **pending**
- Border and instance rules confirmed by: **pending**
- Approver and approval date: **pending**

Dataset selection, commands and comparison protocol are in
[ANNOTATION_PILOT.md](ANNOTATION_PILOT.md). Outstanding limitations are tracked
in [ANNOTATION_CYCLE_GAPS.md](ANNOTATION_CYCLE_GAPS.md).
