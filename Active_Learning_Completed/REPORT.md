# Active Learning: Digits and Wine

Joanne Mesadieu

## Source and corrections

Based on the supplied active_learning(1).ipynb, a Clay Woolam / scikit-learn BSD example. Its executable code uses LabelSpreading, 40 initial labels, 330 Digits observations, gamma=0.25, max_iter=20 and five fits. The introduction says 10 labels and the plot annotations also assume 10; the actual code starts at 40. This analysis follows the executable value. np.in1d was replaced by np.isin for current NumPy compatibility. True class IDs are never multiplied by ten: the NUMBER of labeled observations changes.

## Task 1: label counts and F1 changes

The original 330-sample Digits pool was rerun with 4 and 40 initial labels; 400 is explicitly infeasible. To compare 4, 40 and 400 fairly, the main experiment uses all Digits observations and holds out 20% for testing (1,437 pool, 360 test). Every budget shares the same split and nested random initial label sets within each seed. Seeds: 0,1,2,3,4. Each round queries the five highest-entropy currently unlabeled samples, obtains their true labels from a simulated oracle, then refits. Reported counts are the counts at model fitting, not the count after acquisition. At round zero the ratios are exact; after four acquisitions the counts are 24,60,420.

```text
         experiment  initial_labels  labeled_mean  unlabeled_mean  macro_f1_mean  test_macro_f1_mean  test_macro_f1_std  delta_test_macro_f1_vs_baseline
             Digits               4           4.0          1433.0        0.18213             0.18265            0.03784                         -0.72578
             Digits              40          40.0          1397.0        0.90774             0.90843            0.03559                          0.00000
             Digits             400         400.0          1037.0        0.98000             0.97820            0.00588                          0.06977
               Wine               4           4.0           138.0        0.59400             0.59134            0.17077                         -0.39327
               Wine              40          40.0           102.0        0.94844             0.98461            0.02294                          0.00000
Wine_feasible10fold               1           1.0           141.0        0.17420             0.17467            0.01095                         -0.66914
Wine_feasible10fold              10          10.0           132.0        0.80066             0.84381            0.17319                          0.00000
Wine_feasible10fold             100         100.0            42.0        0.95200             0.97784            0.03034                          0.13403
```

macro_f1_mean is transductive F1 on the remaining unlabeled training-pool observations. test_macro_f1_mean is inductive F1 on the fixed held-out test observations, which are never supplied to graph fitting or label queries. The held-out score is the stronger basis for comparing budgets because the transductive evaluation set shrinks and changes as difficult points are acquired. Macro F1 includes every known dataset class with zero_division=0, including classes absent from the seed labels. Weighted F1 is also saved. A sample SD is reported across seeds; an SD for a single original reproduction run is N/A.

## Task 2: Wine dataset

First, the original algorithm was rerun on all 178 raw Wine rows with 4 and 40 labels. This literal substitution preserves gamma=0.25 and is saved separately from the controlled Wine analysis. Chemical features use different units and scales, so the controlled analysis standardizes using training-pool features only (142 pool, 36 held-out test). The direct 400-label request remains infeasible; it is not silently capped or duplicated. A supplemental 1/10/100 initial-label experiment supplies a feasible exact tenfold comparison with the same Wine split and scaling. One initial label cannot represent all three classes; its poor initial class coverage is intentional and recorded.

## Interpreting the results

Digits: 4 labels: held-out macro F1 0.1826 ± 0.0378; 40 labels: held-out macro F1 0.9084 ± 0.0356; 400 labels: held-out macro F1 0.9782 ± 0.0059.

Change from the lowest to highest feasible initial budget: +0.7955 F1 (+79.55 percentage points). More labels may improve coverage and predictions, but changes need not be monotonic for each seed or acquisition round.

Wine: 4 labels: held-out macro F1 0.5913 ± 0.1708; 40 labels: held-out macro F1 0.9846 ± 0.0229.

Change from the lowest to highest feasible initial budget: +0.3933 F1 (+39.33 percentage points). More labels may improve coverage and predictions, but changes need not be monotonic for each seed or acquisition round.

Wine_feasible10fold: 1 labels: held-out macro F1 0.1747 ± 0.0110; 10 labels: held-out macro F1 0.8438 ± 0.1732; 100 labels: held-out macro F1 0.9778 ± 0.0303.

Change from the lowest to highest feasible initial budget: +0.8032 F1 (+80.32 percentage points). More labels may improve coverage and predictions, but changes need not be monotonic for each seed or acquisition round.

## Limitations

0 fitted models emitted convergence warnings at the original max_iter=20; these fits are flagged in the results instead of being silently altered. Graph quality depends on scaling and gamma. Very few labels can omit classes; results from one random seed are unreliable. The experiments use a simulated oracle, not human annotation. Active acquisition can change the difficulty of the remaining unlabeled subset. Five seeds describe variability but do not establish a universal performance guarantee. Wine raw versus standardized results change geometry as well as scale and must be identified separately.

## Files

f1_results_all_runs.csv contains every fit and explicit infeasible settings. f1_summary_mean_std.csv contains per-round aggregates. initial_budget_comparison.csv contains the primary factor-of-ten comparison and change relative to baseline. classification_reports.json contains seed-zero reports and confusion matrices. queried_labels.csv records oracle queries. f1_label_budget_comparison.png visualizes held-out learning curves.

## Reproduce

Install numpy, pandas, scipy, scikit-learn, matplotlib and threadpoolctl. Run active_learning_analysis.py or all cells in Active_Learning_Completed.ipynb. No external dataset download is required.

## Attribution

Original example: Clay Woolam, BSD license; scikit-learn LabelSpreading active-learning demonstration. Adaptation adds budget comparisons, Wine, held-out evaluation, repeat seeds, result exports and corrected label-count annotations.