"""
export_tables.py - Generates LaTeX and Markdown tables for every result in Stage 3.

Produces:
  - results/tables_latex.tex
  - results/tables_markdown.md
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def generate_all_tables(results_dir: Path) -> tuple[Path, Path]:
    latex_path = results_dir / "tables_latex.tex"
    md_path = results_dir / "tables_markdown.md"

    latex_sections = []
    md_sections = []

    latex_sections.append("% NASA C-MAPSS Predictive Maintenance - Thesis Results Tables\n% Generated from code-produced CSVs.\n")
    md_sections.append("# NASA C-MAPSS Experimental Results Tables\n\nEvery number in these tables is loaded directly from code-produced CSV files in `results/`.\n")

    # -----------------------------------------------------------------------
    # Table 1: RUL Regression Summary over 5 Seeds
    # -----------------------------------------------------------------------
    df_rul = pd.read_csv(results_dir / "rul_seeds_summary.csv")
    latex_sections.append(r"""
% Table 1: RUL Regression Summary (5 Seeds)
\begin{table}[htbp]
\centering
\small
\caption{Prognostic RUL Regression Performance across 5 Random Seeds (Mean $\pm$ Std)}
\label{tab:rul_regression_summary}
\begin{tabular}{llcccc}
\toprule
\textbf{Subset} & \textbf{Model} & \textbf{RMSE} & \textbf{RMSE (clip 125)} & \textbf{MAE} & \textbf{NASA Score} \\
\midrule
""" + "\n".join([
        f"{r['Subset']} & {r['Model']} & {r['RMSE']} & {r['RMSE_clip']} & {r['MAE']} & {r['NASA_Score']} \\\\"
        for _, r in df_rul.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 1: RUL Regression Performance (5 Seeds, Mean ± Std)\n\n" + df_rul.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 2: Anomaly Detection Out-of-Fold Summary
    # -----------------------------------------------------------------------
    df_ano = pd.read_csv(results_dir / "anomaly_metrics_summary.csv")
    latex_sections.append(r"""
% Table 2: Anomaly Detection Fleet Evaluation (5 Seeds)
\begin{table}[htbp]
\centering
\footnotesize
\caption{Out-of-Fold Fleet Anomaly Detection Performance (5 Seeds, Shuffled GroupKFold)}
\label{tab:anomaly_metrics_oof}
\begin{tabular}{llccccc}
\toprule
\textbf{Subset} & \textbf{Model} & \textbf{PR-AUC} & \textbf{F1 Score} & \textbf{Det. Rate (\%)} & \textbf{Mean Lead Time} & \textbf{FA / 100 Healthy} \\
\midrule
""" + "\n".join([
        f"{r['Subset']} & {r['Model'].replace('_', ' ')} & {r['PR_AUC']} & {r['F1']} & {r['Detection_Rate']} & {r['Mean_Lead_Time']} & {r['FA_Per_100_Healthy']} \\\\"
        for _, r in df_ano.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 2: Anomaly Detection Out-of-Fold Fleet Performance (5 Seeds)\n\n" + df_ano.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 3: Official Test Windows Anomaly Detection
    # -----------------------------------------------------------------------
    df_test_ano = pd.read_csv(results_dir / "test_anomaly_metrics.csv")
    latex_sections.append(r"""
% Table 3: Official Test Set Window-Level Anomaly Detection
\begin{table}[htbp]
\centering
\small
\caption{Official Test Engines Window-Level Anomaly Classification (Identical Test Windows)}
\label{tab:test_anomaly_metrics}
\begin{tabular}{llccccc}
\toprule
\textbf{Subset} & \textbf{Model} & \textbf{Precision} & \textbf{Recall} & \textbf{F1 Score} & \textbf{PR-AUC} & \textbf{Prevalence} \\
\midrule
""" + "\n".join([
        f"{r['Subset']} & {r['Model'].replace('_', ' ')} & {float(r['Precision']):.3f} & {float(r['Recall']):.3f} & {float(r['F1']):.3f} & {r['PR_AUC']} & {float(r['Prevalence'])*100:.1f}\\% \\\\"
        for _, r in df_test_ano.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 3: Official Test Engines Window-Level Classification (Identical Windows)\n\n" + df_test_ano.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 4: Lead Time at Controlled False Alarm Budgets
    # -----------------------------------------------------------------------
    df_budgets = pd.read_csv(results_dir / "lead_time_at_fa_budgets.csv")
    latex_sections.append(r"""
% Table 4: Lead Time at Controlled False Alarm Budgets
\begin{table}[htbp]
\centering
\small
\caption{Mean Warning Lead Time (Cycles) at Controlled False Alarm Budgets}
\label{tab:lead_time_fa_budgets}
\begin{tabular}{llccc}
\toprule
\textbf{Subset} & \textbf{Method} & \textbf{FA = 0.1 / 100 Cyc} & \textbf{FA = 0.5 / 100 Cyc} & \textbf{FA = 1.0 / 100 Cyc} \\
\midrule
""" + "\n".join([
        f"{r['Subset']} & {r['Method'].replace('_', ' ')} & {r['LeadTime_at_FA_0.1']} & {r['LeadTime_at_FA_0.5']} & {r['LeadTime_at_FA_1.0']} \\\\"
        for _, r in df_budgets.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 4: Lead Time at Controlled False Alarm Budgets (Cycles before Failure)\n\n" + df_budgets.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 5: Label Sensitivity Sweep (Circularity Proof)
    # -----------------------------------------------------------------------
    df_sweep = pd.read_csv(results_dir / "label_sensitivity_sweep.csv")
    latex_sections.append(r"""
% Table 5: Label Sensitivity Sweep (Circularity Proof)
\begin{table}[htbp]
\centering
\small
\caption{Label Sensitivity Sweep ($N \in \{20, 30, 50, 75, 100\}$): Supervised Lead Time Tracks $N$}
\label{tab:label_sensitivity_sweep}
\begin{tabular}{lccccccc}
\toprule
\textbf{Subset} & \textbf{Cutoff $N$} & \textbf{Prev. (\%)} & \textbf{F1} & \textbf{PR-AUC} & \textbf{Mean Lead Time} & \textbf{Premature (\%)} & \textbf{FA / 100 Cyc} \\
\midrule
""" + "\n".join([
        f"{r['Subset']} & {r['N_Cutoff']} & {r['Prevalence_Pct']} & {r['F1']} & {r['PR_AUC']} & {r['Mean_Lead_Time']} & {r['Premature_Alarm_Pct']} & {r['FA_Per_100_Healthy']} \\\\"
        for _, r in df_sweep.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 5: Label Sensitivity Sweep (Proof of Supervised Lead Time Circularity)\n\n" + df_sweep.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 6: Statistical Significance (Wilcoxon Signed-Rank Tests)
    # -----------------------------------------------------------------------
    df_wilc = pd.read_csv(results_dir / "wilcoxon_tests.csv")
    latex_sections.append(r"""
% Table 6: Wilcoxon Signed-Rank Tests with Holm Correction
\begin{table}[htbp]
\centering
\small
\caption{Wilcoxon Signed-Rank Statistical Tests ($M=8$ Total Hypotheses, Holm Correction)}
\label{tab:wilcoxon_tests}
\begin{tabular}{lllcccc}
\toprule
\textbf{Subset} & \textbf{Metric} & \textbf{Comparison} & \textbf{Raw $p$} & \textbf{Holm $p$} & \textbf{Median Diff.} & \textbf{Bootstrap 95\% CI} \\
\midrule
""" + "\n".join([
        f"{r['Subset']} & {r['Metric'].replace('_', ' ')} & {r['Comparison'].replace('_', ' ')} & {float(r['Raw_p_value']):.4f} & {float(r['Holm_p_value']):.4f} & {float(r['Median_Diff']):+.1f} & {r['CI_95']} \\\\"
        for _, r in df_wilc.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 6: Statistical Significance: Wilcoxon Signed-Rank Tests (Holm Corrected)\n\n" + df_wilc.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 7: Health Index Monotonicity Comparison
    # -----------------------------------------------------------------------
    df_hi = pd.read_csv(results_dir / "health_index_spearman_comparison.csv")
    latex_sections.append(r"""
% Table 7: Health Index Monotonicity
\begin{table}[htbp]
\centering
\small
\caption{Monotonicity Comparison of Health Indices: Spearman Rank Correlation ($\rho$) with True RUL}
\label{tab:health_index_spearman}
\begin{tabular}{lccc}
\toprule
\textbf{Subset} & \textbf{Spearman $\rho$ (PCA-PC1)} & \textbf{Spearman $\rho$ (Mahalanobis Distance)} & \textbf{Superior Metric} \\
\midrule
""" + "\n".join([
        f"{r['Subset']} & {r['Spearman_PCA_PC1']} & {r['Spearman_Mahalanobis']} & Mahalanobis Distance \\\\"
        for _, r in df_hi.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 7: Health Index Monotonicity Comparison (Spearman Rank Correlation with RUL)\n\n" + df_hi.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 8: Cross-Condition Generalization vs In-Domain
    # -----------------------------------------------------------------------
    df_cross = pd.read_csv(results_dir / "cross_condition_transfer.csv")
    latex_sections.append(r"""
% Table 8: Cross-Condition Generalization vs In-Domain (5 Seeds)
\begin{table}[htbp]
\centering
\footnotesize
\caption{Cross-Condition Generalization vs In-Domain Baseline (5 Seeds, Mean $\pm$ Std)}
\label{tab:cross_condition_generalization}
\begin{tabular}{llcccccc}
\toprule
\textbf{Pair} & \textbf{Type} & \textbf{RUL RMSE} & \textbf{RUL RMSE (clip)} & \textbf{Anomaly F1} & \textbf{PR-AUC} & \textbf{$\Delta$ RMSE} & \textbf{$\Delta$ F1 Drop} \\
\midrule
""" + "\n".join([
        f"{r['Pair']} & {r['Pair_Type'].replace('_', ' ')} & {r['RUL_RMSE_Mean']:.2f} $\\pm$ {r['RUL_RMSE_Std']:.2f} & {r['RUL_RMSE_clip_Mean']:.2f} $\\pm$ {r['RUL_RMSE_clip_Std']:.2f} & {r['Anomaly_F1_Mean']:.3f} & {r['Anomaly_PRAUC_Mean']:.3f} & {r['Delta_RMSE_vs_InDomain']:+.2f} & {r['Delta_F1_vs_InDomain']:+.3f} \\\\"
        for _, r in df_cross.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 8: Cross-Condition Generalization vs In-Domain (5 Seeds, Mean ± Std)\n\n" + df_cross.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 9: Split Conformal Prediction Intervals
    # -----------------------------------------------------------------------
    df_conf = pd.read_csv(results_dir / "conformal_prediction_results.csv")
    latex_sections.append(r"""
% Table 9: Split Conformal Prediction Intervals (90% Target Coverage)
\begin{table}[htbp]
\centering
\small
\caption{Split Conformal Prediction Intervals (90\% Target Coverage) Calibrated on Train Engines}
\label{tab:conformal_prediction}
\begin{tabular}{lcccccc}
\toprule
\textbf{Subset} & \textbf{Model} & \textbf{Target Cov.} & \textbf{Quantile $q$} & \textbf{Empirical Cov.} & \textbf{Nominal Width ($2q$)} & \textbf{Clipped Width} \\
\midrule
""" + "\n".join([
        f"{r['Subset']} & {r['Model']} & {float(r['Target_Coverage'])*100:.0f}\\% & {float(r['Conformal_Quantile_q']):.2f} & {float(r['Empirical_Coverage_Test'])*100:.1f}\\% & {float(r['Mean_Interval_Width_Nominal']):.2f} & {float(r['Mean_Interval_Width_Clipped']):.2f} \\\\"
        for _, r in df_conf.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 9: Split Conformal Prediction Intervals on Official Test Engines (90% Target Coverage)\n\n" + df_conf.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 10: Explainability & Sensor Overlap
    # -----------------------------------------------------------------------
    df_overlap = pd.read_csv(results_dir / "shap_sensor_overlap.csv")
    latex_sections.append(r"""
% Table 10: Explainability: TreeSHAP vs Static Baseline Overlap
\begin{table}[htbp]
\centering
\small
\caption{Sensor Attribution Overlap between TreeSHAP and Static Threshold Baseline}
\label{tab:shap_sensor_overlap}
\begin{tabular}{lcccc}
\toprule
\textbf{Subset} & \textbf{Top 3 Overlap} & \textbf{Top 5 Overlap} & \textbf{Top 10 Overlap} & \textbf{Shared Top 10 Sensors} \\
\midrule
""" + "\n".join([
        f"{r['Subset']} & {r['Top3_Overlap_Count']}/3 & {r['Top5_Overlap_Count']}/5 & {r['Top10_Overlap_Count']}/10 ({int(r['Top10_Overlap_Count'])*10}\\%) & {r['Top10_Overlap_Sensors']} \\\\"
        for _, r in df_overlap.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 10: Sensor Attribution Overlap: TreeSHAP vs Static Threshold Baseline\n\n" + df_overlap.to_markdown(index=False) + "\n\n")

    # -----------------------------------------------------------------------
    # Table 11: Preprocessing & Representation Ablation Studies
    # -----------------------------------------------------------------------
    df_abl = pd.read_csv(results_dir / "ablations_summary.csv")
    latex_sections.append(r"""
% Table 11: Preprocessing and Representation Ablations (FD001, 3 Seeds)
\begin{table}[htbp]
\centering
\small
\caption{Ablation Study on Preprocessing, Window Length, and Dimensionality Reduction (FD001, 3 Seeds)}
\label{tab:ablation_studies}
\begin{tabular}{llccccc}
\toprule
\textbf{Category} & \textbf{Configuration} & \textbf{Smoothing} & \textbf{$W$} & \textbf{RUL RMSE} & \textbf{Anomaly F1} & \textbf{PR-AUC} \\
\midrule
""" + "\n".join([
        f"{r['Category']} & {r['Configuration'].replace('_', ' ')} & {r['Smoothing']} & {r['Window_Length']} & {r['RUL_RMSE_Mean']:.2f} $\\pm$ {r['RUL_RMSE_Std']:.2f} & {r['Anomaly_F1_Mean']:.3f} & {r['Anomaly_PRAUC_Mean']:.3f} \\\\"
        for _, r in df_abl.iterrows()
    ]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    md_sections.append("## Table 11: Preprocessing and Representation Ablations (FD001, 3 Seeds)\n\n" + df_abl.to_markdown(index=False) + "\n\n")

    # Save files
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_sections))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_sections))

    print(f"[Saved LaTeX Tables] -> {latex_path}")
    print(f"[Saved Markdown Tables] -> {md_path}")
    return latex_path, md_path


if __name__ == "__main__":
    generate_all_tables(Path(__file__).parent.parent / "results")
