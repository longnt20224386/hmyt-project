# Progress Test 2 — N03 — pet_ct_segmentation

- Fork: https://github.com/longnt20224386/hmyt-project · Branch: gd5-proposed-model
- Project: Development of a semi-automatic lesion segmentation tool for PET/CT images.
- Members (student ID or alias) and roles:
  - 20224386 — Dataset search, slide preparation, and code testing.
  - 20224388 — Dataset search, slide preparation, and code testing.
  - 20233973 — Dataset search, slide preparation, and code testing.
- Status: Progress report based on the supplied slides, five-paper review spreadsheet, source code and local technical tests. Independent multi-patient performance evaluation is not complete. Literature additions and report drafting were AI-assisted. Proposed future dates and task allocations below require team confirmation.
- Evidence layout: keep this report beside the `code/` and `evidence/` folders supplied with it. No medical images are included in the submission package.

## A. Literature review and research gaps

### A1. Search strategy
| Source (PubMed, Scopus, IEEE Xplore, Google Scholar…) | Keywords / query | Year range | No. of hits | No. kept |
|---|---|---|---|---|
| Existing team spreadsheet, PET_CT_Literature_Review.xlsx | Original search queries were not recorded | Included papers: 2005–2024 | Not recorded; cannot reconstruct historical search counts | 5 existing records, #1–5 |
| AI-assisted web discovery of publisher, PMC and arXiv records | PET CT lesion segmentation nnU-Net autoPET 2023 2024 Dice paper | Recent papers prioritized; no formal year filter | Database-wide count unavailable from the search interface; this is a targeted narrative search, not a systematic review | 3 additional records, #8–10 |
| AI-assisted web discovery of HECKTOR papers | HECKTOR 2022 challenge PET CT segmentation results Dice paper | Recent papers prioritized; no formal year filter | Database-wide count unavailable | 2 additional records, #6–7 |

Inclusion / exclusion criteria:
- Include studies on PET/CT lesion segmentation, threshold-based delineation, relevant annotated datasets, or reproducible segmentation benchmarks.
- Prefer 2022–2026 publications. Retain the older thresholding study because it directly motivates the baseline.
- Exclude duplicate records and studies without a relevant segmentation or quantitative delineation task.
- Papers #8–10 are arXiv challenge-method preprints; peer-review status is not assumed. Confirm instructor acceptance of preprints.
- The five added records were selected from an AI-assisted search; the team has not yet documented independent full-text verification of every entry. Values for #1–3 were transcribed from the supplied spreadsheet and still require source checking before final submission.

### A2. Literature matrix (≥ 10 papers, preferably from the last 3–5 years)
| # | Paper (authors, year) | DOI / link | Dataset (n, source) | Method | Main result (metric, value) | External validation? (yes / no) | Main limitation |
|---|---|---|---|---|---|---|---|
| 1 | Gatidis et al., 2022. A whole-body FDG-PET/CT Dataset with manually annotated Tumor Lesions | [DOI](https://doi.org/10.1038/s41597-022-01718-3) | 1,014 studies / 900 patients; TCIA FDG-PET-CT-Lesions; 501 positive and 513 negative examinations | Annotated dataset and 3D nnU-Net baseline | Spreadsheet reports Dice 0.73 ± 0.23; FP volume 8.1 ± 81.4 mL; FN volume 15.1 ± 80.3 mL; original result details pending recheck | No external cohort documented in the supplied review | Single-center dataset; repeated examinations must not be treated as independent patients |
| 2 | Iwasa et al., 2023. The reproducibility of MTV and TLG of soft tissue tumors calculated by FDG-PET | [DOI](https://doi.org/10.1007/s11604-022-01378-8) | 53 soft-tissue tumor cases, according to the supplied review | SUV 2.5 versus 30% SUVmax; repeatability of MTV/TLG | Spreadsheet reports within-reader differences 0.2–3.5% and correlations >0.84; exact endpoints require full-text recheck | No external cohort documented in the supplied review | Repeatability does not establish segmentation accuracy; different tumor population |
| 3 | Nestle et al., 2005. Comparison of Different Methods for Delineation of 18F-FDG PET–Positive Tissue for Target Volume Definition in Radiotherapy of Patients with Non–Small Cell Lung Cancer | [PubMed](https://pubmed.ncbi.nlm.nih.gov/16085592/) | 25 NSCLC patients, according to the supplied review | Manual, SUV 2.5, 40% SUVmax and background-adapted delineation | Spreadsheet reports mean volumes 157.7, 164.6, 53.6 and 94.7 mL respectively; not Dice scores | No external cohort documented in the supplied review | Small, older radiotherapy study; not whole-body segmentation |
| 4 | Gatidis et al., 2024. Results from the autoPET challenge on fully automated lesion segmentation in oncologic PET/CT imaging | [DOI](https://doi.org/10.1038/s42256-024-00912-9) | autoPET whole-body FDG-PET/CT; source cohort 1,014 studies; challenge test-cohort details require full-text extraction | Multi-team automatic segmentation benchmark | Evaluates Dice and FP/FN volumes; smaller and lower-uptake lesions were more often missed. A single leaderboard score is not extracted here | Not yet verified for the exact result to be cited | Algorithm rankings and scores depend on evaluation cohort and lesion characteristics |
| 5 | Xue et al., 2023. 3D PET/CT tumor segmentation based on nnU-Net with GCN refinement | [DOI](https://doi.org/10.1088/1361-6560/acede6) | MICCAI 2022 autoPET dataset; actual experimental sample count not verified | nnU-Net followed by graph convolution refinement | Abstract reports average Dice improvement of 2.1%; HD95 improvement 6.4 and ASSD improvement 1.7; units and relative/absolute interpretation require full text | Not established from the abstract | Added complexity; reported gains cannot be transferred to a threshold method |
| 6 | Oreiller et al., 2022. Head and neck tumor segmentation in PET/CT: The HECKTOR challenge | [DOI](https://doi.org/10.1016/j.media.2021.102336) | 254 cases: 201 training from four centers, 53 test from an unseen fifth center | PET/CT primary-tumor segmentation challenge | Best mean Dice 0.7591; 3D baseline 0.6610 | Yes: unseen-center test | Oropharyngeal primary tumors; not a whole-body lesion task |
| 7 | Andrearczyk et al., 2023. Overview of the HECKTOR Challenge at MICCAI 2022: Automatic Head and Neck Tumor Segmentation and Outcome Prediction in PET/CT | [DOI](https://doi.org/10.1007/978-3-031-27420-6_1) | 883 cases from nine centers: 524 train / 359 test | Primary-tumor and nodal segmentation challenge | Best aggregated Dice 0.788 | Multicenter held-out test; exact unseen-center composition not verified here | Aggregated Dice differs from mean per-case Dice; head-and-neck scope |
| 8 | Alloula, McGowan and Papiez, 2023. Autopet Challenge 2023: nnUNet-based whole-body 3D PET-CT Tumour Segmentation [preprint] | [arXiv](https://arxiv.org/abs/2309.13675) | autoPET 2023; internal test-set n not established from the abstract | nnU-Net training and post-processing experiments | Internal-test Dice 0.69; FN volume 6.27 mL; FP volume 5.78 mL | No for the cited internal-test result | Internal evaluation; final challenge performance is not inferred |
| 9 | Kalisch et al., 2024. Autopet III challenge: Incorporating anatomical knowledge into nnUNet for lesion segmentation in PET/CT [preprint] | [arXiv](https://arxiv.org/abs/2409.12155) | 1,014 FDG and 597 PSMA studies; tracer-specific experiments | Anatomy-guided, tracer-specific nnU-Net ensembles | Cross-validation Dice: FDG 0.7690; PSMA 0.6133 | No for these cross-validation values | Different tracers and internal evaluation; performance is not a universal external benchmark |
| 10 | Wang, Su and Liu, 2024. Dual channel CW nnU-Net for 3D PET-CT Lesion Segmentation in 2024 autoPET III Challenge [preprint] | [arXiv](https://arxiv.org/abs/2409.07144) | 1,611 available studies; 1,038 positive studies used internally | Residual U-Net with sample reweighting | Preliminary five-study Dice 0.8700 | No for this preliminary result | Very small preliminary set drawn from released data; not final external performance |

### A3. Research gaps
| ID | Gap (1–2 sentences) | Evidence (# in A2) | How your project addresses it |
|---|---|---|---|
| G1 | Threshold selection can change delineated volume and derived measurements; stability alone is not accuracy. | #2, #3 | Compare fixed SUV 2.5 with the implemented two-stage threshold method on identical independently defined ROIs; evaluate overlap and MTV/TLG differences. Planned, not completed. |
| G2 | Internal performance does not establish transfer to other centers or anatomical tasks. | #6, #7, #9 | Use patient-independent evaluation and report scope explicitly. External testing is not available yet. |
| G3 | A single overlap score does not fully describe missed lesions or false-positive uptake. | #4, #8 | Plan Dice plus explicitly defined voxel-error volumes, lesion-level detection metrics, and error examples. Do not equate voxel errors with challenge lesion-component FP/FN metrics. |

These are motivations supported by the cited studies, not claims that the proposed method has solved them or that no previous study has considered them.

### A4. PROBAST+AI appraisal of the most important paper in A2
Paper #: 10. Why this paper: selected as the priority methodological appraisal for evaluation independence, a central unresolved risk in our project; #1 remains the main dataset reference. This is a preliminary four-domain appraisal using the course template, not a completed official signaling-question assessment.

Page references below use the 12-page [arXiv PDF](https://arxiv.org/pdf/2409.07144).

| Domain | Risk of bias (low / high / unclear) | Justification (quote or page number in the paper) |
|---|---|---|
| Participants and data sources | High | p. 3: only positive studies retained internally, limiting assessment on negative cases. |
| Predictors | Unclear | pp. 4–5: preprocessing described, but fold-specific fitting is not demonstrated. |
| Outcome | Unclear | p. 3: expert annotations described; blinding details are insufficient for a low-risk judgment. |
| Analysis | High | pp. 5–6: initially held-out B cases are subsequently incorporated into training; p. 3: preliminary set comes from released data. |

Applicability to your team's problem: relevant to evaluation design, but a learned multitracer model differs from our FDG threshold tool. These judgments concern the reported evaluation, not clinical effectiveness.

## B. Results and timeline

### B1. Current results
The application is a semi-automatic threshold tool, not a trained neural network. CT supports viewing and spatial metadata; the core threshold decisions use SUV and an initial ROI. The source applies SUV >= 2.5, 6-connected component extraction, then SUV >= min(0.41 × component SUVmax, 7) within each retained component. The initial filter still applies. No independently obtained ROI was supplied for a new patient-level baseline-versus-proposed experiment.

| Model | Evaluation set (split method, n) | Main metric (value, 95% CI if available) | Secondary metrics | Source of numbers (file / notebook / W&B) |
|---|---|---|---|---|
| Baseline 1 (Milestone 2): SUV >= 2.5 inside a prespecified ROI | Independent evaluation not completed; one case inspected technically; no patient split | Not available — baseline output has not been evaluated independently | Not available | [Source code](../src/segmentation_14.py), lines 1345–1346; [test results](../results/test_results.json) |
| Proposed model (Milestone 3): two-stage threshold with 41% SUVmax and cap 7 | One real case loaded; synthetic functional tests performed; no independent clinical performance cohort | Not available — functional success is not Dice accuracy | Refine requires an ROI; one known-answer synthetic test passed | [Source code](../src/segmentation_14.py), lines 1320–1427; [test harness](../src/test_segmentation.py); [test results](../results/test_results.json) |
| Exploratory comparison of two existing masks, not attributable to either model above | One supplied study; existing candidate versus SEG; provenance and ROI scope unconfirmed | Dice 0.254209; no population CI from n=1 | IoU 0.145612; voxel recall 0.145612; voxel precision 1.000000 | [JSON](../results/test_results.json), key `existing_mask_overlap_unverified_provenance` |

For the existing-mask comparison, every nonzero label was treated as foreground. Reference = 91,462 voxels; candidate = 13,318 voxels; intersection = 13,318 voxels. Thus Dice = 2×13,318/(91,462+13,318). Reference volume = 1,137.880622 mL; candidate volume = 165.689512 mL. Voxel-wise FP volume = 0 and voxel-wise FN volume = 972.191110 mL. These are geometric disagreement measurements, not diagnoses or proof that the application missed this volume. A limited-scope candidate and a whole-body reference would make the comparison unfair.

Technical findings from the actual source:
- GUI initialization was tested offscreen; this was not a complete manual usability test.
- CT/PET loaded at size 400×400×327; spacing approximately 2.03642×2.03642×3 mm. Inspected images and masks shared geometry.
- PET maximum = 97,630.109375; SUV maximum = 44.363022. PET/SUV ratio was approximately 2,200.709 on tested nonzero voxels. The application does not convert PET to SUV; supplying raw PET to the SUV input invalidates the intended threshold interpretation. Conversion provenance still requires checking.
- A 27-voxel synthetic ROI with background SUV 3 and one center voxel at 20 retained exactly one voxel, as expected.
- Changing the global control from 10% to 60% produced identical output; the global algorithm hardcodes 41%.
- A synthetic 1,000-component case failed at label 256 because the output array is uint8.
- Code inspection shows SUVpeak returns SUVmax; do not claim independently measured SUVpeak.

### B2. Ablation (Milestone 3)
| Configuration (component removed / changed) | Main metric | Difference vs full model | Source |
|---|---|---|---|
| Remove relative-threshold stage; retain SUV 2.5 baseline | Not available — planned independent experiment | Not available | Proposed protocol in B1; implementation lines 1345–1346 |
| Remove absolute cap 7 from relative-threshold stage | Not available — planned independent experiment | Not available | Proposed change to lines 1379–1381; not yet executed |
| Global UI control 10% versus 60%, synthetic ROI | Outputs identical | No change in this synthetic test | [Test results](../results/test_results.json) |

The final row is a control-wiring test, not evidence of a clinically successful ablation. Patient-level ablation remains unfinished.

### B3. Comparison with the literature
- #1: the supplied review gives baseline Dice 0.73 ± 0.23. Our existing-mask Dice 0.254209 comes from one case with unknown candidate provenance. Dataset membership, ROI scope, averaging and method independence differ; no performance ranking is justified.
- #6: mean Dice 0.7591 was measured on 53 unseen-center head-and-neck cases. Our one-case mask comparison is neither an external test nor the same anatomical task.
- #8: internal-test Dice 0.69 concerns automatic whole-body nnU-Net segmentation. Our tool needs an ROI and has no independent cohort result. Also, our FP/FN values count erroneous voxels, whereas autoPET metrics may count whole unmatched lesion components. Verify metric definitions before numerical comparison.

### B4. Timeline to completion
The dates below are proposed planning dates, not confirmed course deadlines. All members share the work; assignments remain joint because the team has not specified individual leads.

| Task | Owner (student ID / role) | Milestone (M3 week 10 · M4 week 12 · Final week 15) | Planned date | Status (Not started / In progress / Done + evidence) |
|---|---|---|---|---|
| Prepare ten-paper matrix and review draft | 20224386 / 20224388 / 20233973 | M3 week 10 | 2026-11-10 | In progress — ten records in A2; team verification pending |
| Technical source/data tests | Joint code-testing role | M3 week 10 | 2026-11-10 | Done for the limited tests described in B1 — evidence/test_results.json |
| Resolve mask provenance, ROI scope and SUV source | Joint data role | M3 week 10 | 2026-11-12 | Not started — unresolved B5 questions |
| Fix label dtype and threshold-control behavior; correct SUVpeak reporting | Joint code-testing role | M4 week 12 | 2026-11-17 | Not started — original tested code retained |
| Run patient-independent baseline/proposed/ablation experiments | All three student IDs | M4 week 12 | 2026-11-24 | Not started — protocol requires confirmation |
| Complete reporting checklist, sample-size limitations and error analysis | All three student IDs | Final week 15 | 2026-12-08 | Not started |
| Final report and slides | All three student IDs | Final week 15 | 2026-12-15 | In progress — preliminary slides exist; final evidence pending |

### B5. Risks / blockers / questions for the instructor
- Is a rule-based semi-automatic segmentation tool acceptable as the proposed model, or must a learned baseline/model be added?
- Were the existing candidate masks drawn independently or derived from SEG? Were they restricted to neck/thorax? This must be resolved before interpreting overlap.
- One inspected case cannot establish generalization; repeated scans must stay within the same patient split.
- Raw PET/SUV scale confusion and the observed software defects can change results.
- The exact local data release, download date, access route and privacy checks remain unverified.
- Are arXiv challenge-method preprints acceptable in the ten-paper review?
- Please confirm the appropriate reporting checklist for deterministic ROI-based segmentation and how the AUC/calibration requirement should be handled without probability outputs.

## C. Self-assessment against the FAIR, TRIPOD+AI, PROBAST+AI appendix

### C1. FAIR — data
| Item | Answer | Evidence |
|---|---|---|
| Source, version, download date; checksum of the training file | Source attributed to FDG-PET-CT-Lesions in supplied slides. Local release and original download date unknown. No model training occurred. SHA-256 values computed for the five inspected files; these identify local bytes, not the original release. | [Checksums](evidence/data_checksums.json); [TCIA collection](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/) |
| Licence / DUA; basis for being allowed to use the data | Not yet verified for these local files. Current TCIA distinguishes defaced CC BY 4.0 images from controlled originals; autoPET lists an FDAT NIfTI route under CC BY-NC 4.0. Confirm actual source and terms rather than assigning a licence by filename. | [TCIA](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/); [autoPET data page](https://autopet.org/fdgpetct.html) |
| Datasheet: population, inclusion/exclusion criteria, variable definitions and units, limitations | Partial description: FDG oncology dataset; one study inspected. Inputs: resampled CT and SUV; label: binary nonzero mask. CT units and SUV conversion need provenance checks. MTV in mL; TLG = MTV × mean SUV. Cohort inclusion criteria and independent ROI rules are not finalized. | Slides; B1; code/segmentation_14.py lines 1776–1800. A standalone datasheet is not complete. |
| No patient identifiers | Not yet verified. Matching geometry and numeric-array checks are not a de-identification audit. No medical images are included in this package; full local paths and image headers must be reviewed before wider sharing. | Submitted package contains code, aggregate results and checksums, not image volumes. Privacy review remains pending. |

### C2. Reporting guideline
- Your research question: Within independently specified ROIs on FDG PET/CT, how does the two-stage SUV threshold method compare with fixed SUV 2.5 in reference-mask overlap and MTV/TLG differences?
- Matching reporting guideline (TRIPOD+AI / STARD-AI / DECIDE-AI) and why (Appendix §3.2): Not finalized. The current artifact is deterministic segmentation software, not a trained prognostic model or a prospective clinical deployment study. Instructor confirmation against Appendix §3.2 is required before claiming checklist compliance; selecting a label merely because AI is mentioned would be inappropriate.
- Checklist items met / total; items still missing: Not assessed / denominator not established until the applicable guideline is selected. Missing items include a prespecified evaluation cohort, independent ROI/reference protocol, sample-size rationale, patient-level results and uncertainty, and completed item-by-item checklist.
- Outcome, label, prediction time T0, variables available at T0: Intended outcome is the lesion mask and derived MTV/TLG. SEG is a candidate reference whose provenance must be confirmed. T0 is after CT/SUV loading and independent user ROI selection, before Refine. Reference labels must not supply the input ROI for an independent accuracy claim. If reference masks are intentionally refined, describe the task as reference-conditioned mask refinement instead.

### C3. PROBAST+AI — self-appraisal of your project
Preliminary domain judgments follow the course table; they do not substitute for the full official signaling questions.

| Domain | Risk of bias (low / high / unclear) | Evidence in the repo (file, line) | What you will do to reduce the risk |
|---|---|---|---|
| Participants and data sources | High | evidence/test_results.json: one inspected study; no cohort manifest or split | Define eligibility, include negative cases as appropriate, expand the cohort and group scans by patient |
| Predictors | High | code/segmentation_14.py:1202–1207 reads intensities directly; 1345–1381 applies SUV thresholds; test JSON shows different PET/SUV scales | Validate SUV conversion, geometry and ROI provenance; freeze parameters before test evaluation |
| Outcome (watch for incorporation bias) | Unclear | code/segmentation_14.py:1253–1270 loads SEG into the working mask; 1330–1346 can use that mask as input. Whether this happened for the existing candidate is unknown | Retain an immutable independent reference; record operator input; distinguish reference-conditioned refinement from independent prediction |
| Analysis (patient-level split, fit on train only, sample size, missing data) | High | evidence/test_results.json has n=1 mask comparison; code/segmentation_14.py:1364 uses uint8; 1366 fixes threshold percentage | Fix defects; document failures; tune only on development data; evaluate held-out patients; compute patient-level uncertainty when sufficient data exist |

Applicability (population, setting where the model would be used): Educational/research prototype for operator-assisted FDG PET/CT segmentation. No evidence supports clinical deployment, all tumor types, other tracers, or other centers. Intended neck-to-thorax scope from the slides must be aligned with the evaluation reference.

### C4. Quick checklist (Appendix §7)
| Item | Met / Not yet | Evidence |
|---|---|---|
| Datasheet | Not yet | Partial data description in C1; standalone datasheet incomplete |
| Data version + checksum | Not yet — checksum done, version unresolved | evidence/data_checksums.json |
| No identifiers; basis for data use | Not yet | Access-route and privacy review pending; C1 |
| Correct reporting guideline + completed checklist | Not yet | C2; instructor clarification required |
| Outcome, label, T0 defined | Not yet — intended definitions stated, provenance unresolved | C2 |
| Patient-level split; preprocessing fit on train only | Not yet | No evaluation split; no learned preprocessing in this tested script, but future tuning must stay in development data |
| Logistic regression baseline (or a suitable baseline, with reason) | Not yet — suitable baseline specified, evaluation pending | SUV 2.5 is a direct rule-based segmentation comparator; logistic regression is not implemented. B1 |
| AUC + calibration (curve, in-the-large, slope) + Brier, with 95% CI | Not yet — applicability requires clarification | Current output is a hard mask, not calibrated probabilities. Do not manufacture these metrics or a population CI from one case; see B5/C2 |
| Is the sample size sufficient; applicability limits | Not yet — n=1 is insufficient for performance claims | B1 and C3 explicitly state the limitation |

## D. AI use declaration
| Tool | Used for which part | How the team verified the output |
|---|---|---|
| ChatGPT / Codex | Explain assignment and GitHub workflow; inspect slides and spreadsheet; draft this English report and proposed timeline | Team verification pending: compare against the instructor template, actual responsibilities and course calendar |
| ChatGPT / Codex with web search | Add papers #6–10 and assist with methodological appraisal | Primary publisher/arXiv links and PDF page locations provided; students must independently verify the entries, especially #1–3 transferred from the spreadsheet |
| Codex with Python / NumPy / SimpleITK / PyQt5 | Execute limited functional tests, compare existing masks, compute SHA-256 checksums | Machine-readable outputs and portable test harness supplied. Student rerun/review not yet confirmed. No baseline/proposed accuracy values were invented |

The tested original source is included unchanged as code/segmentation_14.py. Its SHA-256 is recorded in evidence/test_results.json. No fixes or successful clinical evaluation are claimed. The 10% versus 60% experiment and the label-overflow case are software tests, not patient-level ablations.
