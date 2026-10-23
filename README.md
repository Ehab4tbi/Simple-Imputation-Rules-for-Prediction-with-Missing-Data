# Simple Imputation Rules for Prediction with Missing Data: Contrasting Theoretical Guarantees with Empirical Performance

*Paper summary — Bertsimas, Delarue, and Pauphilet, Transactions on Machine Learning Research (2024), arXiv 2104.03158*

# Part I. Paper summary

## 1. Main problem addressed
*(Paper: Abstract, Section 1)*

The paper studies prediction, not inference, when some input features are missing. In practice, prediction with missing data is usually done with an impute-then-regress pipeline: missing entries are filled in first, and a model is then trained on the completed data. Because this workflow is borrowed from inference, the missing-at-random (MAR) assumption and “plausible” imputations are often seen as requirements. The paper asks which imputation rules actually lead to good predictions, whether MAR is needed for that, and whether simple rules such as mean or mode imputation are good enough compared with sophisticated methods such as MICE.

## 2. Proposed solution
*(Paper: Section 2, Theorem 2.1, Corollary 2.2)*

The main contribution is a theoretical characterisation of impute-then-regress pipelines, followed by an extensive empirical check. In a setting where one feature X₁ can be missing and is replaced by a deterministic value µ(x₂:d) (which covers mean, mode, and conditional mean or mode imputation), Theorem 2.1 gives the prediction rule that a universally consistent learner converges to with infinite data. When the imputed value appears, the learner predicts a weighted average of the outcome for “truly missing” points and for observed points that happen to take the imputed value. The weight α(x) is the posterior probability that X₁ was originally missing, given that it now equals the imputed value.

Corollary 2.2 follows: the pipeline is Bayes optimal if and only if α(x) = 1 (or both outcomes coincide), meaning the model can always tell imputed values from real ones and “de-impute” them. The authors summarise this as **“crude is good”**: for prediction, a good imputation encodes missingness as conspicuously as possible rather than producing plausible values. The result does not require MAR. It also supports the common practice of adding the missingness mask as extra features.

## 3. Optimizations and special cases
*(Paper: Sections 3.2, 3.3, and 4; Theorem 4.1, Examples 4.2 and 4.3)*

* **Categorical features (Section 3.2).** Mode imputation is asymptotically *inconsistent*: the mode is a value that observed data also take with positive probability, so α(x) < 1. Encoding missingness as a separate category gives α(x) = 1 and is consistent, so it should be preferred.
* **Continuous features (Section 3.3).** If X₁ has a continuous conditional distribution, an observed value equals the imputed mean with probability zero, so α(x) = 1 and mean imputation is asymptotically *consistent*. This generalises an earlier result of Josse et al. (2019). Sophisticated imputation is therefore not needed for consistency and may even hurt in practice.
* **MAR versus NMAR (Section 4, Theorem 4.1).** For two missingness mechanisms with the same missing rate, the theorem gives an exact condition for when non-random missingness yields lower optimal prediction error than MAR. Example 4.2 shows NMAR helping (missingness reveals the outcome), and Example 4.3 shows MAR helping. The usual “MAR good, NMAR bad” view therefore does not carry over to prediction.
* **Relation to Le Morvan et al. (2021).** Theorem 2.1 applies in a more restrictive one-missing-feature setting than Le Morvan et al.’s “almost all imputations” result, but it gives an explicit condition, α(x), that explains *why* consistency holds or fails.

## 4. Strengths and limitations
*(Paper: Sections 2.2, 3.2, 3.3, and 4)*

**Strengths.** The theory is simple and gives a clear, actionable rule: prefer imputations the model can recognise, such as a new category for categorical features and the mean for continuous ones. It needs no MAR assumption and covers several imputation families at once. The empirical study is broad, combining synthetic, semi-real, and real data with paired statistical tests, and the authors report openly where the evidence disagrees with the theory.

**Limitations.** The theory is asymptotic: it assumes infinite data and a universally consistent learner, so it says nothing about finite samples or simple parametric models. It is proven for a single missing feature, and the extension to several is only claimed. On real outcomes the effects are not statistically significant, so practical support comes mainly from synthetic signals, and the effect sizes are small (often in the second decimal of R² or AUC). For continuous features, only mean imputation and MICE are compared. The authors also show that the best imputation depends on the downstream model (Figure 2) and question whether synthetic missing-data benchmarks are realistic.

## 5. Evaluation methodology
*(Paper: Section 3.1, Tables 1 and 2, Figures 1 to 3, Appendix C)*

* **Synthetic data (Appendix C.1).** There are 10 Gaussian features with a low-rank-plus-noise covariance (rank 5), training sizes from 40 to 1,000, and 5,000 test rows. The outcome is a linear function or a random two-layer neural network of 5 features, with a signal-to-noise ratio of 2. Missingness is either MCAR or censoring of values above a percentile (a form of NMAR), at 8 missing rates from 0.1 to 0.8.
* **Semi-real and real data (Appendix C.2).** The study uses 63 public UCI and RDatasets datasets with their real feature values and real missingness patterns. For semi-real instances, synthetic outcomes are generated under MAR, NMAR, or adversarial missingness; real outcomes are used where available. Each dataset is evaluated over 10 train/test splits.
* **Pipelines and metrics.** Missing categorical features are mode-imputed or given their own category, and missing continuous features are mean- or MICE-imputed. The downstream model is chosen by 5-fold cross-validation among a regularised linear model, a decision tree, a random forest, and XGBoost. Out-of-sample R² (or AUC for binary outcomes) is compared with paired t-tests and Wilcoxon tests.
* **Main results.**
  * **Mode imputation (Table 1):** it significantly reduces accuracy in all synthetic-signal settings, but the effect is not significant on real outcomes.
  * **MICE vs mean imputation (Table 2):** on fully synthetic data, MICE is better under MCAR but clearly worse under censoring. On semi-real and real data, MICE is significantly worse in 5 of 7 cases and not significantly different in the other 2.
  * **Effect of missing rate (Figure 1):** MICE is preferable at low missing rates and mean imputation at high rates, with an earlier break-even point under censoring.
  * **Downstream models (Figures 2 and 3):** the preferred imputation depends on the model; MICE is better with linear regression, mean imputation with random forests. XGBoost’s built-in missing-value handling behaves differently depending on the missingness pattern.
