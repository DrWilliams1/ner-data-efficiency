# NER experiment review

## Recorded results

The completed Colab experiment used a Tesla T4. All nine validation runs and the selected model's test result were recalculated from the supplied word-level predictions and probabilities.

| Training fraction | Sentences | Mean entity F1 ± SD | Mean precision | Mean recall | Mean training time ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10% | 1,404 | 0.4679 ± 0.0644 | 0.4586 | 0.4782 | 15.60 ± 0.39 s |
| 25% | 3,510 | 0.8241 ± 0.0096 | 0.8261 | 0.8221 | 37.68 ± 0.28 s |
| 100% | 14,041 | 0.9312 ± 0.0005 | 0.9327 | 0.9297 | 152.11 ± 3.89 s |

Each row summarizes seeds 42, 43, and 44 on the same 3,250 validation sentences. SD is the sample standard deviation across those runs; it is not a confidence interval. Timing covers the synchronized training call, excluding loading, validation, and saving.

The full-data fraction had the highest mean validation F1 and was selected. Its prespecified seed-42 checkpoint achieved **0.9030 test entity F1**, precision **0.9024**, and recall **0.9037** on **3,453 sentences**. This is one checkpoint's test result, not a three-seed test average.

The test set contains 5,648 reference entities. The model produced 5,656 entities: **5,104 matched exactly**, **552 were unmatched predictions**, and **544 reference entities were missed**. An incorrect type or boundary can contribute both an unmatched prediction and a missed reference; these counts are not numbers of incorrect sentences.

## What the comparison shows

Increasing the training fraction from 10% to 25% raised mean validation F1 by **0.3561**. Moving from 25% to 100% added **0.1071 F1**, while increasing mean training-call time from **37.68 to 152.11 seconds**—about **4.04 times as long**. All three seeds improved at each increase in training fraction.

The 25% run provides an observed lower-time option at lower entity F1. Full-data training is the best performer under the declared highest-F1 rule. The experiment does not identify a universally optimal cost/performance point or show that more data always produces these gains.

Variation across seeds was much larger at 10% than at 100%: F1 ranged from **0.4217 to 0.5415** at 10%, compared with **0.9307 to 0.9315** at 100%. The seed changes both the sample and the training process for smaller fractions, so these sources of variation are not separately estimated.

The selected checkpoint's validation F1 was **0.9315**, compared with **0.9030** on test, a difference of **0.0285**. The difference alone does not identify its cause or prove overfitting.

## Test performance by entity type

| Type | Reference entities | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| PER | 1,617 | 0.9569 | 0.9604 | 0.9586 |
| LOC | 1,668 | 0.9252 | 0.9197 | 0.9224 |
| ORG | 1,661 | 0.8956 | 0.8784 | 0.8869 |
| MISC | 702 | 0.7480 | 0.7949 | 0.7707 |

PER (people) had the highest test F1; MISC (miscellaneous entities) had the lowest. Those descriptive scores depend on the benchmark's labeling rules and support counts. They are not results for every kind of person, place, organization, or miscellaneous entity in other domains.

## Selected validation errors

Five validation rows were reviewed after numerical verification. These examples were chosen to illustrate distinct disagreements, not sampled to estimate how common each cause is. Their observations are in [error_analysis.csv](error_analysis.csv). The original 811-row error worksheet remains unchanged.

- **Row 2653 — partial boundary:** the reference entity includes “The” in “The Netherlands,” while the model starts at “Netherlands.” The type is correct, but strict scoring requires the complete span.
- **Row 14 — missed modifier:** a hyphenated descriptor is labeled MISC but predicted as non-entity. Person names in the same sentence are recognized.
- **Row 756 — organization and label transition:** a lower-case organization phrase is missed, and a separate person-name token receives an invalid continuation tag after punctuation. These are observable errors; their internal causes are not established.
- **Rows 2911 and 2881 — possible reference ambiguity:** a dateline-like item is labeled PER but predicted LOC, and an ordinary noun is labeled MISC but predicted non-entity. Both merit annotation review. No ground-truth labels were changed, and these examples do not establish that the reference annotations are wrong.

The worksheet's score is the mean of maximum word-level softmax scores across the listed span. A missed entity can have a high score because the model strongly predicts `O`. It is not a calibrated probability that a complete entity is correct. The test predictions contain 94 invalid `I` transitions, which strict BIO2 scoring does not silently repair.

## Method and trade-offs

- **Data and model:** CoNLL-2003 from `lhoestq/conll2003`, using 14,041 training, 3,250 validation, and 3,453 test sentences; `google-bert/bert-base-cased` with a newly initialized nine-label token classifier. Both source revisions are fixed and recorded.
- **Comparison:** nested 10%, 25%, and 100% sentence samples within each of seeds 42, 43, and 44. Every run starts from the pretrained model, and initial classifier hashes match across fractions within each seed. All model parameters are trained.
- **Training:** one epoch, batch size 16, learning rate 0.00002, weight decay 0.01, linear schedule, no warmup, float32. The recorded numbers of optimizer steps are 88, 220, and 878 for the respective fractions.
- **Scoring:** one label per original word, sentence boundaries retained, strict IOB2 exact entity matches, and per-type counts. Continuation subwords and special/padding tokens are ignored with `-100`. Inputs are not silently truncated.
- **Selection:** highest mean validation entity F1, with smaller fraction as the exact-tie rule. Seed 42 was fixed in advance for the selected fraction's final test; the highest-scoring seed was not chosen after observing results.

One epoch gives larger samples more optimizer updates. The comparison therefore measures data size together with its one-epoch training budget, not the isolated causal effect of additional data at equal compute. Hyperparameters were not tuned separately for the fractions. Three seeds describe limited variation; no confidence interval or significance claim is made.

The official splits were preserved. The exact-sentence audit found **130** normalized unique sentences shared by train/validation, **78** by train/test, and **26** by validation/test. Within-split duplicate-row counts were **1,360**, **182**, and **273**, respectively. These are exact-text counts, not counts of duplicate entities or an audit of semantic duplicates or pretraining exposure. They limit a claim that every evaluation sentence is wholly unseen.

Softmax scores are uncalibrated. News-domain performance does not establish performance on policy, maritime, or operational documents. No independent annotation reliability study was conducted.
