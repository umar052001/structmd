# Research priorities after 0.3.0

These are evaluation/design references, not proposed runtime dependencies.
Sources were checked against their authors' repositories on 2026-11-02.
Web search was unavailable; this is a focused shortlist, not an exhaustive review.

## 1. OmniDocBench (CVPR 2025)

- Paper: <https://arxiv.org/abs/2412.07626>
- Authors' evaluation code: <https://github.com/opendatalab/OmniDocBench>
- Why it matters: annotated text, reading order, layout regions, tables and
  formulas expose errors that our five-paper text-layer proxy cannot measure.
- Small next experiment: pin a dataset revision and select a fixed annotated
  subset spanning papers, financial reports, scans and multi-column pages.
  Export page predictions and use the official evaluator rather than calling
  our current overlap score an OmniDocBench score.
- Keep the reference annotations unchanged when matching predictions. Record
  matching/evaluator versions; changed matching alone can change scores.

## 2. GriTS: Grid Table Similarity Metric for Table Structure Recognition

- Paper: <https://arxiv.org/abs/2203.12555>
- Authors' implementation: <https://github.com/microsoft/table-transformer>
- Why it matters: equal row widths say nothing about correct cell contents,
  row/column alignment or merged-cell structure. GriTS evaluates cell topology,
  content and location separately.
- Small next experiment: manually label a fixed set of our difficult tables
  and measure content/topology separately. Our `List[List[str]]` representation
  cannot express row/column spans; do not pretend width consistency covers them.

## 3. PubTables-1M (CVPR 2022)

- Paper: <https://openaccess.thecvf.com/content/CVPR2022/html/Smock_PubTables-1M_Towards_Comprehensive_Table_Extraction_From_Unstructured_Documents_CVPR_2022_paper.html>
- Authors' dataset/model code: <https://github.com/microsoft/table-transformer>
- Why it matters: canonicalization and explicit header/cell annotations make
  table structure evaluation meaningful rather than rewarding arbitrary splits.
- Small next experiment: use a bounded held-out subset to identify whether our
  failures come from detection, text recognition or schema limitations before
  changing the VLM prompt or adding another inference model.

## Decision gate

First establish labeled failures and a reproducible competitor baseline. Then
try one prompt/schema/model change at a time on a development split, keeping a
separate held-out split. Track quality, cold inference latency and peak memory;
do not select a change solely because it improves the five cached papers.
