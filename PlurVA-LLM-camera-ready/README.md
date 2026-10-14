# PlurVA-LLM 2026 Camera-Ready Package

Camera-ready (de-anonymized) version of the StealthRL paper, accepted as a long archival paper at the PlurVA-LLM 2026 workshop (AACL-IJCNLP 2026) via ACL Rolling Review commitment. Derived from `../acl_arr/` (the anonymous ARR submission).

## Build

```bash
tectonic -X compile stealthrl.tex
```

## Check

```bash
uvx --from git+https://github.com/acl-org/aclpubcheck aclpubcheck --paper_type long stealthrl.pdf
pdffonts stealthrl.pdf   # all fonts should be embedded
```

## Changes from the ARR submission

- `\usepackage[final]{acl}` (no line numbers, no page numbers).
- Author block, public code and model links, and Acknowledgments restored; anonymous 4open and demo links removed (no demo link in camera-ready).
- Training framework de-anonymized (Tinker API, Thinking Machines).
- Bibliography: published venues for peer-reviewed works, official ACL Anthology entries (with DOIs) for ACL venues, arXiv DOIs for preprints; capitalization protected.
- `acl_natbib.bst` updated to the current upstream acl-org/acl-style-files version.
- No changes to paper content in response to reviews (separate step).
