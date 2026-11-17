# Attribution and changes

The NER learning baseline comes from Chapter 11 of *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst (2024).

- Original chapter: https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter11
- Original code copyright: 2024 Jay Alammar & Maarten Grootendorst.
- Upstream code license: Apache-2.0; the full license and notice are retained in `LICENSE`.
- License source: https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/LICENSE

This separate portfolio adaptation changes the experiment and evaluation: nested data fractions; three seeds; sentence-preserving strict entity evaluation; one label per original word; no silent truncation; validation-driven fraction selection; a fixed seed for final testing; synchronized training timing; saved word-level probabilities and gold labels for both splits; an error worksheet; plots; and offline verification.

The supplied source notebook is retained separately and is not overwritten or bundled. Its saved score is not presented as a result of this new experiment. Joe Williams supplied the completed Colab result archive on November 17, 2026. The nine validation runs and selected-model final test were verified from saved predictions; no training was performed during packaging.

## Model and data

- Model: https://huggingface.co/google-bert/bert-base-cased
- Model revision: `cd5ef92a9fb2f889e972770a36d4ed042daf221e`.
- Dataset: https://huggingface.co/datasets/lhoestq/conll2003
- Dataset revision: `19edcb426bfd625c275c17b9a99b2239243f4377`.
- Original shared task: https://aclanthology.org/W03-0419/
- CoNLL-2003 citation: Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition.
- Token-classification guide: https://huggingface.co/docs/transformers/en/tasks/token_classification
- Strict scoring reference: https://github.com/chakki-works/seqeval

The model card lists Apache-2.0. The dataset repository does not provide a clear license declaration in its displayed card metadata. The code license does not relicense dataset content. The package excludes raw corpus text and model checkpoints.

## Completed-run packaging

Added measured results, timing comparisons, per-type interpretation, and five validation-row observations. All 44 supplied result files remain byte-identical. The original error worksheet is unchanged; observations are in a separate analysis file. The executed notebook retains all 16 original code cells, execution counts, numerical outputs, tables, and chart. Result summaries and interpretation were added after execution. Colab-specific progress/download widgets and table-interaction metadata were removed for GitHub display. No model was retrained during packaging.
