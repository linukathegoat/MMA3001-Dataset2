"""Run the complete analysis and save auditable outputs and figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sensor_project import (SENSOR_ID, SENSOR_ROOM, METHODS,
                                load_temperature, evaluate,
                                choose_by_validation_mae)


def save_plots(df, results, predictions, example, winner, fig_dir):
    """Export report-ready figures with units and descriptive labels."""
    plt.style.use("default")
    # Full month raw observations
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(df["timestamp_utc"],df["temperature_c"], lw=0.65)
    ax.set(title="Observed temperature — sensor " + SENSOR_ID,
           xlabel="Timestamp (UTC)",ylabel="Temperature (°C)")
    fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(fig_dir/"01_observed_temperature.png",dpi=180);plt.close(fig)

    # Sample artificial gap in independent holdout month portion.
    fig,ax=plt.subplots(figsize=(10,4.6))
    xs=pd.to_datetime(example["times"],unit="s",utc=True)
    ts=pd.to_datetime(example["hidden_times"],unit="s",utc=True)
    ax.plot(xs,example["actual"],"o-",color="black",lw=1.2,markersize=3,
            label="Original measured temperature (truth)")
    ax.axvspan(ts[0],ts[-1],color="grey",alpha=0.12,label="Masked interval")
    for method in METHODS:
        ax.plot(ts,example["predictions"][method],marker="x",lw=1.2,label=method)
    ax.set(title="Example of reconstructing six hidden sensor readings (holdout)",
           xlabel="Time (UTC)",ylabel="Temperature (°C)")
    ax.legend(fontsize=8,ncol=2);fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(fig_dir/"02_example_missing_gap.png",dpi=180);plt.close(fig)

    # Per-gap validation results
    val=results[results["split"]=="validation"]
    fig,ax=plt.subplots(figsize=(9,5))
    widths=0.19
    gaps=sorted(val.gap_readings.unique())
    for j,method in enumerate(METHODS):
        vals=[val[(val.method==method)&(val.gap_readings==g)].MAE_C.iloc[0] for g in gaps]
        ax.bar(np.arange(len(gaps))+(j-1.5)*widths,vals,widths,label=method)
    ax.set(xticks=np.arange(len(gaps)),xticklabels=[str(g) for g in gaps],
           xlabel="Consecutive readings intentionally hidden per gap",
           ylabel="Validation MAE (°C)",title="Sensitivity to gap length (validation period)")
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(fig_dir/"03_validation_gap_sensitivity.png",dpi=180);plt.close(fig)

    # Independent holdout: 6-reading gap selection fixed before test
    hold=results[(results.split=="holdout") & (results.gap_readings==6)]
    fig,ax=plt.subplots(figsize=(8,4.5))
    labels=hold.method.tolist();values=hold.MAE_C.tolist()
    bars=ax.bar(labels,values)
    for b,label in zip(bars,labels):
        if label==winner:b.set_edgecolor("black");b.set_linewidth(2.0)
    ax.set(title="Independent late-May holdout: six-reading missing gaps",
           ylabel="Holdout MAE (°C)")
    fig.tight_layout();fig.savefig(fig_dir/"04_holdout_comparison.png",dpi=180);plt.close(fig)

    # Residuals selected winner for independent holdout.
    e=predictions[(predictions.split=="holdout") &(predictions.gap_readings==6)
                  &(predictions.method==winner)].error_c.to_numpy()
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.hist(e,bins=15,edgecolor="black",linewidth=0.4)
    ax.axvline(0,color="black",lw=1,linestyle="--")
    ax.set(title=f"Holdout prediction errors — selected method: {winner}",
           xlabel="Estimated minus measured temperature (°C)",ylabel="Number of predictions")
    fig.tight_layout();fig.savefig(fig_dir/"05_holdout_error_distribution.png",dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5))
    run=val.groupby("method")["median_prediction_runtime_ms"].median().reindex(METHODS)
    ax.bar(run.index,run.values)
    ax.set(title="Median prediction call runtime (validation cases)",
           ylabel="Wall-clock runtime (ms)")
    fig.tight_layout();fig.savefig(fig_dir/"06_runtime_comparison.png",dpi=180);plt.close(fig)


def write_results_summary(audit, results, winner, dest):
    """Produce a plain-language summary containing only measured, computed values."""
    val=results[results.split=="validation"]
    hold=results[(results.split=="holdout")&(results.gap_readings==6)]
    lines=["# Actual numerical results — MMA3001 Dataset 2",
           "",f"Sensor: `{SENSOR_ID}` — {SENSOR_ROOM}.",
           "", "## Data audit",
           f"- Study interval: {audit['study_start_utc']} to {audit['study_end_exclusive_utc']} (UTC, exclusive end)",
           f"- Original messages from this sensor in entire export: {audit['sensor_rows']:,}",
           f"- Temperature values in May study interval: {audit['temperature_values_in_range']:,}",
           f"- Distinct May readings: **{audit['unique_measurements']:,}**",
           f"- Repeated same-timestamp messages removed: {audit['repeated_messages_removed']:,}",
           f"- Conflicting same-timestamp readings: {audit['conflicting_duplicates']}",
           f"- Median original sampling interval: {audit['median_observed_interval_minutes']:.2f} min",
           f"- Observed temperature range: {audit['minimum_temperature_c']:.1f} to {audit['maximum_temperature_c']:.1f} °C",
           "", "## Validation (chronologically earlier 70% of May data)", "",
           "Masked known readings, then predicted without training on those hidden targets.",
           "The following table gives validation MAE in °C. Blocks of 3, 6 and 12 readings",
           "were tested separately using 15 non-overlapping blocks per gap size.","",
           val.pivot(index="method",columns="gap_readings",values="MAE_C").to_markdown(floatfmt=".4f"),
           "",f"**Selected method, based only on validation MAE: `{winner}`.**",
           "", "## Independent later-May holdout (last 30%, six-reading gaps)", "",
           hold[["method","n_hidden_readings","mean_gap_duration_minutes","MAE_C","RMSE_C","MaxAE_C","within_0_5_C_pct","median_prediction_runtime_ms"]].to_markdown(index=False,floatfmt=".4f"),
           "", "## Important interpretation limits", "",
           "- All withheld values were real measured readings; the missingness was simulated, not naturally missing ground truth.",
           "- Testing only one sensor and month does not demonstrate performance elsewhere.",
           "- No method may extrapolate past the first/last retained observation.",
           "- Adjacent readings are correlated, so hidden predictions are not fully independent.",
           "- Temperature values could contain sensor bias or calibration errors; no physical ground-truth instrument was available.",
           "- Timing uses a local process and includes estimator fitting; it will vary by computer.",
           "- Source CSV filename references May–December 2024 but individual JSON times also include other years; this project explicitly filters **May 2024 timestamps**.",
           "", "## Reproducibility", "", "Run `python run_project.py --input <environment-zip-path>` from project root.",
           "Check generated CSV files, PNGs and `pytest -q`. Do not upload raw proprietary or private sensor logs to a public repository without permission.",""]
    (dest/"REAL_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description="MMA3001 temperature gap reconstruction analysis")
    parser.add_argument("--input",required=True,help="Path to original environmental CSV ZIP")
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    figs=root/"figures";out=root/"results"
    figs.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    full,audit=load_temperature(args.input)
    # Strict chronological separation: last 30% is held out until model choice.
    n=int(0.70*len(full));validation=full.iloc[:n].copy();holdout=full.iloc[n:].copy()
    val_summary,val_errors,_=evaluate(validation,split_name="validation",n_blocks=15,seed=2026)
    winner=choose_by_validation_mae(val_summary)
    # Model selection is finished BEFORE evaluating the independent later-May portion.
    hold_summary,hold_errors,example=evaluate(holdout,split_name="holdout",
                                              gap_lengths=(6,),n_blocks=15,seed=8227)
    results=pd.concat([val_summary,hold_summary],ignore_index=True)
    errors=pd.concat([val_errors,hold_errors],ignore_index=True)
    audit["validation_points"] = int(len(validation))
    audit["holdout_points"] = int(len(holdout))
    audit["chosen_method_validation_only"] = winner
    (out/"data_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    results.to_csv(out/"method_metrics.csv",index=False)
    errors.to_csv(out/"prediction_errors.csv",index=False)
    # Cleaned time-series is for local-only processing, not public upload.
    full.to_csv(out/"cleaned_temperature_LOCAL_ONLY.csv",index=False)
    save_plots(full,results,errors,example,winner,figs)
    write_results_summary(audit,results,winner,out)
    print("Study sensor:",SENSOR_ID,SENSOR_ROOM)
    print("Unique real temperature observations:",len(full))
    print("Selected method (validation only):",winner)
    print(results[["split","gap_readings","method","MAE_C","RMSE_C","median_prediction_runtime_ms"]].to_string(index=False))
    print("All results saved under:",out)


if __name__=="__main__":
    main()
