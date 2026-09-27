# sle-periop-gc-analysis

Analysis code for a head-to-head cohort study of **early intensive-care-unit
glucocorticoid dose and 30-day mortality in systemic lupus erythematosus (SLE)
versus rheumatoid arthritis (RA)**, with the underlying rheumatic diagnosis
treated as an effect modifier.

Primary estimand: the disease × dose interaction on 30-day mortality, fitted by
Firth penalised likelihood. The headline estimate is intentionally not quoted
here — it will be added together with the citation once the article is
published.

**This repository contains code only.** No patient-level data are included.
All source databases are governed by the PhysioNet data-use agreement (DUA);
users must obtain credentialed access and run the extraction scripts locally.

---

## Data sources

| Database | Role in this study |
|---|---|
| MIMIC-IV v3.1 | Derivation cohort (`mimiciv_icu`, `mimiciv_hosp` schemas) |
| eICU-CRD v2.0 | External rheumatic-ICU cohort (`eicu` schema) |
| NWICU (Northwestern ICU) | Secondary external cohort (`nwicu` schema) |
| INSPIRE | Screened for external replication; no eligible admissions |

Discharge narrative text is read from `mimiciv_note.discharge` (one note per
hospital admission). Databases must be loaded into a local PostgreSQL instance.

## Requirements

- Python ≥ 3.10, PostgreSQL ≥ 14
- `psycopg2-binary`, `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`,
  `scikit-learn`, `Pillow`
- No R dependency in this repository

### Database configuration

Credentials are **not** stored in the code. Every database script reads:

| Variable | Default |
|---|---|
| `MIMIC_DB_HOST` | `localhost` |
| `MIMIC_DB_PORT` | `5432` |
| `MIMIC_DB_USER` | `postgres` |
| `MIMIC_DB_PASSWORD` | *(empty)* |

```bash
export MIMIC_DB_HOST=localhost
export MIMIC_DB_USER=postgres
export MIMIC_DB_PASSWORD='<your password>'
```

## Repository layout

```
analysis/     numbered pipeline: schema probes -> extraction -> analysis -> measurement validation
```

Working directories are created by `00_init.py` and are **not** versioned:

```
<repo root>/
  data/       extracted cohorts and derived annotation corpora (git-ignored)
  out/        result tables (CSV), logs (TXT) and figures (PNG) (git-ignored)
  out/fig/
  sql/        reserved for standalone SQL (this study's SQL is inline in Python)
```

Every script derives its paths from its own location:

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "out")
```

so the repository can be cloned anywhere.

## Getting started

```bash
python analysis/00_init.py            # create data/, out/, out/fig/
python analysis/01_probe_gc_sources.py   # optional: verify the GC prescription source
python analysis/112_extract_mimic_harmonized.py
python analysis/151_extract_rheum_cohort.py   # -> data/cohort_rheum_icu.csv
python analysis/152_head_to_head.py           # -> out/table_head2head_*.csv
```

## Pipeline map

Numbers are the run order, not a version chain; scripts whose number is
followed by `a`/`b` are tuning probes for the script that follows.

### 0. Environment and schema probes

| Script | Purpose |
|---|---|
| `00_init.py` | Create `data/`, `out/`, `out/fig/` |
| `01_probe_gc_sources.py`, `02_probe_dose_format.py`, `03_probe_covars.py` | Inventory glucocorticoid prescription sources, dose units and candidate covariates |
| `100_probe_eicu_schema.py`, `101_probe_external_detail.py`, `102_probe_nwicu_eicu2.py`, `103_probe_covariates_ext.py` | External-database schema and coverage probes |
| `11_diag_joins.py` | Diagnose row inflation in the diagnosis joins |
| `70_probe_activity.py`, `71_probe_windows.py` | Activity-marker availability; exposure-window definitions |
| `150_probe_rheum_taxonomy.py`, `160_probe_external_rheum.py` | Rheumatic-disease taxonomy and external feasibility |

### 1. Cohort extraction

| Script | Output |
|---|---|
| `10_extract_cohort.py`, `12_extract_cohort_v2.py`, `13_extract_cohort_v3.py` | Early SLE-only cohort iterations, superseded by `151`. Their `OUT` points into `data/` exactly as in the original run; retained for provenance. |
| `112_extract_mimic_harmonized.py` | `data/cohort_mimic_harmonized.csv` — MIMIC-IV cohort on the transportable variable set |
| `110_extract_eicu.py` | `data/cohort_eicu.csv` |
| `111_extract_nwicu.py` | `data/cohort_nwicu.csv` |
| `21_extract_negctrl.py` | `data/negctrl_vars.csv` — negative-control exposure variables |
| `151_extract_rheum_cohort.py` | `data/cohort_rheum_icu.csv` — **head-to-head SLE/RA cohort** (n = 1,636) |
| `154_extract_abx_outcomes.py` | `data/abx_outcomes_rheum.csv` — antibiotic/culture timeline |
| `161_extract_external_rheum.py` | `data/cohort_eicu_rheum.csv`, `data/cohort_nwicu_rheum.csv` |
| `72_extract_fixes.py` | `data/cohort_v4_fixes.csv` — post-extraction corrections |
| `120_external_validation.py` | SLE-only external validation |

Key exposure/covariate definitions from `151_extract_rheum_cohort.py`:

- **Landmark** — ICU length of stay ≥ 24 h, first ICU stay per hospital
  admission, age ≥ 18 years
- **Exposure** — prednisone-equivalent glucocorticoid **dose intensity during
  the first 24 h after ICU admission** (window = min(24 h, ICU LOS)),
  stratified as 0 / >0–<10 / 10–<50 / ≥50 mg/day. This is a dose-intensity
  measure, not cumulative dose
- **Disease labels** — SLE: ICD-10 `M32*` / ICD-9 `7100*`; RA: ICD-10 `M05*`/`M06*`
  / ICD-9 `714*` with adult-onset Still disease (`M06.1`) removed
- **Covariates** — age, sex, vasopressor ≤ 24 h, mechanical ventilation ≤ 24 h,
  renal/respiratory/hepatic failure, shock, coagulopathy, organ-dysfunction
  count, SOFA, surgical-service admission, procedure count,
  non-glucocorticoid immunosuppression, hydroxychloroquine

### 2. Narrative-annotation corpora

| Script | Output |
|---|---|
| `158_extract_gc_indication.py` | `data/gc_indication.csv` — whether the discharge narrative states an explicit GC indication |
| `159_extract_gc_history.py` | `data/gc_history.csv` — GC history / chronic-use statements |
| `163_extract_inactivity.py` | `data/gc_inactivity.csv` — statements of disease inactivity |
| `168_activity_rule_engine.py` | `data/gc_activity_density.csv` — three-rule engine plus a continuous narrative-density score |
| `168a_probe_v8.py`, `168b_probe_tune.py` | Threshold tuning probes for the rule engine |

### 3. Head-to-head analysis and robustness

| Script | Purpose |
|---|---|
| `152_head_to_head.py`, `153_head_to_head_robustness.py` | Interaction and robustness battery (v4) |
| `155_head_to_head_v5.py`, `156_head_to_head_v5_robustness.py` | Primary-outcome specification (v5) |
| `20_analysis_main.py` | Main analysis driver |
| `30_iptw_sensitivity.py` | Design-weighted IPTW |
| `40_msm_timedep.py` | Marginal structural model with time-dependent exposure |
| `80_positive_control_fix.py` | Positive control: GC → hyperglycaemia at 48 h |
| `81_detection_bias.py` | Surveillance-intensity / detection-bias checks |
| `82_robustness.py` | Additional sensitivity analyses |
| `160_indication_mechanism.py` | Indication-stratified mechanism analysis, `spec` vs `gen` decomposition |
| `164_inactivity_mechanism.py` | Decomposition by documented disease inactivity |
| `162_external_head_to_head.py` | External replication of the interaction |
| `204_density_interaction.py` | Narrative density, length adjustment, multiplicity and the interaction (v11) |

### 4. Measurement validation of the text phenotype

These scripts document the rule-engine development programme, including the
iterations that failed and were retired.

| Script | Purpose |
|---|---|
| `165_make_blind_sheets.py` | Build blinded double-coding sheets |
| `166_double_coding_metrics.py` | Inter-rater metrics on the blinded set |
| `167_adjudication_estimates.py` | Re-weight stratified estimates by adjudicated labels |
| `169_v8_validation.py` | In-sample / out-of-sample validation of the rule engine |
| `170_blind_sample_v9.py` | Out-of-sample sampling frame (v9) |
| `171_replication_metrics.py`, `172_replication_tests.py` | Replication metrics and McNemar tests |
| `200_rule_ablation.py`, `201_ablation_tests.py` | Rule ablation, with/without each rule |
| `207_manual_v2_sheet.py`, `208_manual_v2_impact.py` | Revised manual and its impact on performance |
| `212_human_adjud_prep.py`, `213_human_adjud_analysis.py` | Human adjudication material and analysis (prespecified thresholds) |

### 5. Figure source scripts

`50_figures.py`, `83_figures_v2.py`, `130_figures_v3.py`, `170_figures_v4.py`,
`180_figures_v5.py`, `190_figures_v6.py`, `192_figures_v7.py`,
`194_figures_v7b.py`, `196_figures_v8.py`, `198_figures_v9.py`,
`202_figures_v10.py`, `205_figures_v11.py`, `209_figures_v12.py` — each draws
the single-panel PNGs that the manuscript figure plates are composed from.

## What this repository deliberately does not contain

- **Patient-level data.** All `data/` and `out/` artefacts are git-ignored.
  The extracted cohorts contain `subject_id` / `stay_id` / `hadm_id` and are
  covered by the PhysioNet DUA.
- **Manuscript production code.** The HTML report builders, the manuscript /
  supplement / cover-letter builders, the shared HTML templating library, the
  reference harvester and the figure-plate compositor are not published, so
  that the unpublished manuscript text is not disclosed.
- **Ad-hoc session helpers.** A handful of one-off scripts used during
  interactive analysis are omitted.

## Reproducibility notes

- Every statistic reported in the manuscript is written to `out/table_*.csv`
  by these scripts and read back into the manuscript by a templating pipeline;
  no number is transcribed by hand.
- Sparse-event models use Firth penalised likelihood; Wald confidence intervals
  and penalised-likelihood-ratio P values are reported together and the
  difference in construction is stated in the table footnotes.
- Multiplicity is handled in a three-stage scheme (primary outcome, then
  secondary family, then the full family) with Benjamini–Hochberg control.
- Text-derived measures are validated on data physically disjoint from the
  rule-engineering sample; positive predictive values are reported instead of
  overall agreement, because the class balance is highly unequal.

## Citation

If you use this code, please cite the accompanying article (in preparation) and
this archive.

| | DOI |
|---|---|
| Concept (always resolves to the latest version) | [10.5281/zenodo.22987549](https://doi.org/10.5281/zenodo.22987549) |
| Version v1.0.0 (archived 2026-09-27) | [10.5281/zenodo.22987550](https://doi.org/10.5281/zenodo.22987550) |

> Wang K. *sle-periop-gc-analysis: analysis code for a head-to-head cohort study
> of early ICU glucocorticoid dose in systemic lupus erythematosus versus
> rheumatoid arthritis* (v1.0.0). Zenodo; 2026.
> https://doi.org/10.5281/zenodo.22987550

Note on versions: the Zenodo **v1.0.0** snapshot is the commit tagged `v1.0.0`.
This `README.md` is a living document, so later edits (this Citation table, for
example) appear here and on GitHub but not inside that frozen archive.

## Licence

MIT — see `LICENSE`.
