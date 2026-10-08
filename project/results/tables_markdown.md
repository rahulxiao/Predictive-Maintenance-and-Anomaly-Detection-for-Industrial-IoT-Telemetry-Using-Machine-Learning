# NASA C-MAPSS Experimental Results Tables

Every number in these tables is loaded directly from code-produced CSV files in `results/`.

## Table 1: RUL Regression Performance (5 Seeds, Mean ± Std)

| Subset   | Model   | RMSE          | RMSE_clip     | MAE           | NASA_Score    |
|:---------|:--------|:--------------|:--------------|:--------------|:--------------|
| FD001    | CNN     | 19.03 +/- nan | 17.81 +/- nan | 15.17 +/- nan | 659.6 +/- nan |
| FD001    | LSTM    | 14.79 +/- nan | 13.46 +/- nan | 11.08 +/- nan | 351.1 +/- nan |
| FD001    | RF      | 17.69 +/- nan | 16.59 +/- nan | 13.28 +/- nan | 790.2 +/- nan |
| FD001    | XGBoost | 14.89 +/- nan | 13.54 +/- nan | 11.16 +/- nan | 332.3 +/- nan |


## Table 2: Anomaly Detection Out-of-Fold Fleet Performance (5 Seeds)

| Subset   | Model               | PR_AUC         | ROC_AUC        | Prevalence   | F1            | FDR           | Detection_Rate   | Mean_Lead_Time   |   Median_Lead_Time | Premature_Alarm_Pct   | FA_Per_100_Healthy   | MatchedFAR_LeadTime   | MatchedFAR_DetectRate   | MatchedFAR_PrematurePct   | MatchedFAR_FA100   |
|:---------|:--------------------|:---------------|:---------------|:-------------|:--------------|:--------------|:-----------------|:-----------------|-------------------:|:----------------------|:---------------------|:----------------------|:------------------------|:--------------------------|:-------------------|
| FD001    | EWMA_Control_Limits | N/A (discrete) | N/A (discrete) | 17.5%        | 0.786 (tuned) | nan           | 98.0%            | 38.6             |                 34 | 2.0%                  | 0.19                 | 38.6                  | 98.0%                   | 2.0%                      | 0.19               |
| FD001    | IsolationForest     | 0.395 +/- nan  | 0.837 +/- nan  | 17.5%        | 0.390 +/- nan | 0.758 +/- nan | 76.0% +/- nan%   | 63.8 +/- nan     |                 61 | 24.0% +/- nan%        | 6.81 +/- nan         | 63.8 +/- nan          | 76.0% +/- nan%          | 24.0% +/- nan%            | 6.81 +/- nan       |
| FD001    | LSTM                | 0.971 +/- nan  | 0.994 +/- nan  | 17.5%        | 0.891 +/- nan | 0.177 +/- nan | 100.0% +/- nan%  | 29.5 +/- nan     |                 28 | 0.0% +/- nan%         | 0.00 +/- nan         | 21.3 +/- nan          | 100.0% +/- nan%         | 0.0% +/- nan%             | 0.00 +/- nan       |
| FD001    | RandomForest        | 0.977 +/- nan  | 0.995 +/- nan  | 17.5%        | 0.913 +/- nan | 0.136 +/- nan | 100.0% +/- nan%  | 29.1 +/- nan     |                 28 | 0.0% +/- nan%         | 0.00 +/- nan         | 21.8 +/- nan          | 100.0% +/- nan%         | 0.0% +/- nan%             | 0.00 +/- nan       |
| FD001    | Static_Threshold    | N/A (discrete) | N/A (discrete) | 17.5%        | 0.841 (tuned) | nan           | 100.0%           | 26.8             |                 24 | 0.0%                  | 0.00                 | 26.8                  | 100.0%                  | 0.0%                      | 0.00               |
| FD001    | XGBoost             | 0.982 +/- nan  | 0.996 +/- nan  | 17.5%        | 0.926 +/- nan | 0.098 +/- nan | 100.0% +/- nan%  | 28.2 +/- nan     |                 28 | 0.0% +/- nan%         | 0.00 +/- nan         | 23.8 +/- nan          | 100.0% +/- nan%         | 0.0% +/- nan%             | 0.00 +/- nan       |


## Table 3: Official Test Engines Window-Level Classification (Identical Windows)

| Subset   | Model               | Split            |   Precision |   Recall |       F1 | PR_AUC         | ROC_AUC        |   Prevalence |
|:---------|:--------------------|:-----------------|------------:|---------:|---------:|:---------------|:---------------|-------------:|
| FD001    | Static_Threshold    | test (truncated) |   0.843478  | 0.584337 | 0.690391 | N/A (discrete) | N/A (discrete) |    0.0325618 |
| FD001    | EWMA_Control_Limits | test (truncated) |   0.500998  | 0.756024 | 0.602641 | N/A (discrete) | N/A (discrete) |    0.0325618 |
| FD001    | IsolationForest     | test (truncated) |   0.0647679 | 1        | 0.121656 | 0.1803         | 0.9235         |    0.0325618 |
| FD001    | RandomForest        | test (truncated) |   0.763224  | 0.912651 | 0.831276 | 0.9346         | 0.9975         |    0.0325618 |
| FD001    | XGBoost             | test (truncated) |   0.768638  | 0.900602 | 0.829404 | 0.9379         | 0.9976         |    0.0325618 |
| FD001    | LSTM                | test (truncated) |   0.723982  | 0.963855 | 0.826873 | 0.9175         | 0.9974         |    0.0325618 |


## Table 4: Lead Time at Controlled False Alarm Budgets (Cycles before Failure)

| Subset   | Method              |   LeadTime_at_FA_0.1 |   LeadTime_at_FA_0.5 |   LeadTime_at_FA_1.0 |
|:---------|:--------------------|---------------------:|---------------------:|---------------------:|
| FD001    | Static_Threshold    |                 46.1 |                 48.4 |                 49.5 |
| FD001    | EWMA_Control_Limits |                 47.1 |                 48.8 |                 49.7 |
| FD001    | Mahalanobis_HI      |                 48.9 |                 49.5 |                 50.3 |
| FD001    | IsolationForest     |                nan   |                nan   |                nan   |
| FD001    | RandomForest        |                 52.5 |                 55   |                 56.5 |
| FD001    | XGBoost             |                 43.6 |                 43.6 |                 43.6 |
| FD001    | LSTM                |                 43.4 |                 44.6 |                 45.9 |


## Table 5: Label Sensitivity Sweep (Proof of Supervised Lead Time Circularity)

| Subset   |   N_Cutoff |   Prevalence_Pct |    F1 |   PR_AUC |   ROC_AUC |   Detection_Rate_Pct |   Mean_Lead_Time |   Median_Lead_Time |   Premature_Alarm_Pct |   FA_Per_100_Healthy |
|:---------|-----------:|-----------------:|------:|---------:|----------:|---------------------:|-----------------:|-------------------:|----------------------:|---------------------:|
| FD001    |         20 |             11.8 | 0.92  |    0.982 |     0.997 |                  100 |             17.8 |               17   |                     0 |                 0    |
| FD001    |         30 |             17.5 | 0.924 |    0.982 |     0.996 |                  100 |             28.6 |               28   |                     0 |                 0    |
| FD001    |         50 |             28.8 | 0.924 |    0.981 |     0.991 |                  100 |             48.8 |               47   |                     0 |                 0    |
| FD001    |         75 |             42.9 | 0.915 |    0.977 |     0.98  |                   90 |             72.8 |               71.5 |                    10 |                 2.38 |
| FD001    |        100 |             57   | 0.895 |    0.97  |     0.954 |                   45 |             86.9 |               90   |                    55 |                25.57 |


## Table 6: Statistical Significance: Wilcoxon Signed-Rank Tests (Holm Corrected)

| Subset   | Metric       | Comparison                  |   Median_Diff |   Mean_Diff | CI_95          |   Raw_p_value |   Total_Tests |   Holm_p_value |   Bonferroni_p_value | Significant_after_Holm   |
|:---------|:-------------|:----------------------------|--------------:|------------:|:---------------|--------------:|--------------:|---------------:|---------------------:|:-------------------------|
| FD001    | False_Alarms | XGBoost vs Static_Threshold |             0 |         0   | [+0.00, +0.00] |     1         |             2 |       1        |             1        | False                    |
| FD001    | Lead_Time    | XGBoost vs Static_Threshold |             2 |         1.4 | [-1.00, +5.00] |     0.0614234 |             2 |       0.122847 |             0.122847 | False                    |


## Table 7: Health Index Monotonicity Comparison (Spearman Rank Correlation with RUL)

| Subset   |   Sensors_Count |   Spearman_PCA_PC1 |   Spearman_Mahalanobis | Best_Index   |
|:---------|----------------:|-------------------:|-----------------------:|:-------------|
| FD001    |              14 |             0.6369 |                 0.7141 | Mahalanobis  |


## Table 8: Cross-Condition Generalization vs In-Domain (5 Seeds, Mean ± Std)

| Source_Domain   | Target_Domain   | Pair         | Pair_Type    |   RUL_RMSE_Mean |   RUL_RMSE_Std |   RUL_RMSE_clip_Mean |   RUL_RMSE_clip_Std |   RUL_NASA_Mean |   RUL_NASA_Std |   Anomaly_F1_Mean |   Anomaly_F1_Std |   Anomaly_PRAUC_Mean |   Anomaly_PRAUC_Std |   Lead_Time_FA05_Mean |   Lead_Time_FA05_Std |   Delta_RMSE_vs_InDomain |   Delta_RMSE_clip_vs_InDomain |   Delta_NASA_vs_InDomain |   Delta_F1_vs_InDomain |   Delta_PRAUC_vs_InDomain |   Delta_LeadTime_vs_InDomain |
|:----------------|:----------------|:-------------|:-------------|----------------:|---------------:|---------------------:|--------------------:|----------------:|---------------:|------------------:|-----------------:|---------------------:|--------------------:|----------------------:|---------------------:|-------------------------:|------------------------------:|-------------------------:|-----------------------:|--------------------------:|-----------------------------:|
| FD001           | FD001           | FD001->FD001 | In_Domain    |         13.2694 |            nan |              12.0154 |                 nan |         253.292 |            nan |          0.828169 |              nan |             0.937292 |                 nan |               20.85   |                  nan |                        0 |                             0 |                        0 |                      0 |                         0 |                            0 |
| FD001           | FD002           | FD001->FD002 | Cross_Domain |         54.6653 |            nan |              44.2795 |                 nan |      193914     |            nan |          0.12061  |              nan |             0.0719   |                 nan |               35.7831 |                  nan |                        0 |                             0 |                        0 |                      0 |                         0 |                            0 |


## Table 9: Split Conformal Prediction Intervals on Official Test Engines (90% Target Coverage)

| Subset   | Model   |   Target_Coverage |   Calibration_Units_Count |   Calibration_Windows_Count |   Test_Engines_Count |   Conformal_Quantile_q |   Empirical_Coverage_Test |   Mean_Interval_Width_Nominal |   Mean_Interval_Width_Clipped |
|:---------|:--------|------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------:|--------------------------:|------------------------------:|------------------------------:|
| FD001    | XGBoost |               0.9 |                        20 |                        3490 |                  100 |                20.4042 |                      0.84 |                       40.8085 |                       36.0836 |


## Table 10: Sensor Attribution Overlap: TreeSHAP vs Static Threshold Baseline

| Subset   | Top10_SHAP_Sensors                           | Top10_StaticBaseline_Sensors                 |   Top3_Overlap_Count | Top3_Overlap_Sensors   |   Top5_Overlap_Count | Top5_Overlap_Sensors   |   Top10_Overlap_Count | Top10_Overlap_Sensors               |
|:---------|:---------------------------------------------|:---------------------------------------------|---------------------:|:-----------------------|---------------------:|:-----------------------|----------------------:|:------------------------------------|
| FD001    | s3, s11, s2, s17, s12, s14, s4, s9, s7, s20  | s11, s4, s12, s7, s15, s21, s20, s2, s9, s14 |                    1 | s11                    |                    2 | s11, s12               |                     8 | s11, s12, s14, s2, s20, s4, s7, s9  |
| FD002    | s11, s4, s17, s15, s2, s14, s6, s3, s9, s12  | s11, s4, s15, s13, s8, s17, s2, s9, s3, s14  |                    2 | s11, s4                |                    3 | s11, s15, s4           |                     8 | s11, s14, s15, s17, s2, s3, s4, s9  |
| FD003    | s11, s3, s9, s12, s17, s14, s4, s8, s7, s13  | s11, s15, s4, s17, s21, s20, s9, s14, s3, s2 |                    1 | s11                    |                    2 | s11, s17               |                     6 | s11, s14, s17, s3, s4, s9           |
| FD004    | s11, s3, s14, s17, s9, s4, s15, s2, s12, s13 | s15, s11, s4, s9, s14, s8, s13, s17, s3, s7  |                    1 | s11                    |                    3 | s11, s14, s9           |                     8 | s11, s13, s14, s15, s17, s3, s4, s9 |


## Table 11: Preprocessing and Representation Ablations (FD001, 3 Seeds)

| Category   | Configuration                     | Smoothing   |   Window_Length | Representation   |   RUL_RMSE_Mean |   RUL_RMSE_Std |   RUL_RMSE_clip_Mean |   RUL_RMSE_clip_Std |   RUL_MAE_Mean |   RUL_MAE_Std |   Anomaly_F1_Mean |   Anomaly_F1_Std |   Anomaly_PRAUC_Mean |   Anomaly_PRAUC_Std |
|:-----------|:----------------------------------|:------------|----------------:|:-----------------|----------------:|---------------:|---------------------:|--------------------:|---------------:|--------------:|------------------:|-----------------:|---------------------:|--------------------:|
| Baseline   | Smoothing_OFF_W30_EngineeredStats | OFF         |              30 | Engineered_Stats |         13.1666 |            nan |              11.9495 |                 nan |        10.0409 |           nan |          0.828169 |              nan |             0.937292 |                 nan |
| Smoothing  | Smoothing_ON_W30_EngineeredStats  | ON          |              30 | Engineered_Stats |         13.5958 |            nan |              12.5484 |                 nan |        10.1158 |           nan |          0.837877 |              nan |             0.940328 |                 nan |

