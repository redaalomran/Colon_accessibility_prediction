import numpy as np
import pandas as pd
import pyBigWig
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shap
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    auc
)

# Generate non-overlapping 1000 bp bins for the human genome (hg38)
def generate_bins(chrom_sizes_path, output_path, bin_size=1000):
    with open(chrom_sizes_path) as f, open(output_path, "w") as out:
        for line in f:
            chrom, size = line.strip().split("\t")
            size = int(size)
            for start in range(0, size, bin_size):
                end = min(start + bin_size, size)
                out.write(f"{chrom}\t{start}\t{end}\n")

# Check if bins already exist to avoid redundant computation
if os.path.exists("data/colon_1000bp_bins.bed"):
    print("colon_1000bp_bins.bed already exists. Skipping generation.")
else:
    print("Generating 1000bp bins from hg38.chrom.sizes...")
    generate_bins("data/hg38.chrom.sizes", "data/colon_1000bp_bins.bed")
    print("Generation complete: colon_1000bp_bins.bed")

# Load bins and restrict to autosomes + chrX
bins_df = pd.read_csv("data/colon_1000bp_bins.bed", sep="\t", names=["chrom", "start", "end"])
wanted_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
bins_df = bins_df[bins_df["chrom"].isin(wanted_chroms)].reset_index(drop=True)
print("Filtered bins:", len(bins_df))

# Load .bigWig signal tracks
bw_H3K27ac = pyBigWig.open("data/ENCFF277XII_H3K27ac.bigWig")
bw_H3K4me3 = pyBigWig.open("data/ENCFF213WKK_H3K4me3.bigWig")
bw_H3K27me3 = pyBigWig.open("data/ENCFF457PEW_H3K27me3.bigWig")
bw_h3k9me3 = pyBigWig.open("data/ENCFF063OHO_H3K9me3.bigWig")
bw_h3k4me1 = pyBigWig.open("data/ENCFF182EJH_H3K4me1.bigWig")
bw_h3k36me3 = pyBigWig.open("data/ENCFF059WYR_H3K36me3.bigWig")
bw_CTCF = pyBigWig.open("data/ENCFF813QCX_CTCF.bigWig")
bw_RAD21 = pyBigWig.open("data/ENCFF027QAE_RAD21.bigWig")
bw_ATAC = pyBigWig.open("data/ENCFF624HRW_ATAC.bigWig")

# Initialize feature storage lists
h3k27ac_vals = []
h3k4me3_vals = []
h3k27me3_vals = []
h3k9me3_vals = []
h3k4me1_vals = []
h3k36me3_vals = []
CTCF_vals = []
RAD21_vals = []
atac_vals = []

# Robust mean extraction with NaN filtering
def safe_mean_signal(bw, chrom, start, end):
    try:
        values = np.array(bw.values(chrom, start, end, numpy=True))
        values = values[~np.isnan(values)]
        return np.mean(values) if len(values) > 0 else 0.0
    except:
        return 0.0

# Extract mean signal per bin for each feature
for i, row in tqdm(bins_df.iterrows(), total=len(bins_df)):
    chrom = row["chrom"]
    start = int(row["start"])
    end = int(row["end"])

    if start < end:
        try:
            h3k27ac_vals.append(safe_mean_signal(bw_H3K27ac, chrom, start, end))
        except:
            h3k27ac_vals.append(0.0)

        try:
            h3k4me3_vals.append(safe_mean_signal(bw_H3K4me3, chrom, start, end))
        except:
            h3k4me3_vals.append(0.0)

        try:
            h3k27me3_vals.append(safe_mean_signal(bw_H3K27me3, chrom, start, end))
        except:
            h3k27me3_vals.append(0.0)

        try:
            h3k9me3_vals.append(safe_mean_signal(bw_h3k9me3, chrom, start, end))
        except:
            h3k9me3_vals.append(0.0)

        try:
            h3k4me1_vals.append(safe_mean_signal(bw_h3k4me1, chrom, start, end))
        except:
            h3k4me1_vals.append(0.0)

        try:
            h3k36me3_vals.append(safe_mean_signal(bw_h3k36me3, chrom, start, end))
        except:
            h3k36me3_vals.append(0.0)

        try:
            CTCF_vals.append(safe_mean_signal(bw_CTCF, chrom, start, end))
        except:
            CTCF_vals.append(0.0)

        try:
            RAD21_vals.append(safe_mean_signal(bw_RAD21, chrom, start, end))
        except:
            RAD21_vals.append(0.0)

        try:
            atac_vals.append(safe_mean_signal(bw_ATAC, chrom, start, end))
        except:
            atac_vals.append(0.0)
    else:
        h3k27ac_vals.append(0.0)
        h3k4me3_vals.append(0.0)
        h3k27me3_vals.append(0.0)
        h3k9me3_vals.append(0.0)
        h3k4me1_vals.append(0.0)
        h3k36me3_vals.append(0.0)
        CTCF_vals.append(0.0)
        RAD21_vals.append(0.0)
        atac_vals.append(0.0)

# Add extracted values as new columns to bins dataframe
bins_df["H3K27ac"] = h3k27ac_vals
bins_df["H3K4me3"] = h3k4me3_vals
bins_df["H3K27me3"] = h3k27me3_vals
bins_df["H3K9me3"] = h3k9me3_vals
bins_df["H3K4me1"] = h3k4me1_vals
bins_df["H3K36me3"] = h3k36me3_vals
bins_df["CTCF"] = CTCF_vals
bins_df["RAD21"] = RAD21_vals
bins_df["ATAC"] = atac_vals

# Save the final signaled bin table
bins_df.to_csv("data/colon_bins.csv", sep="\t", index=False)

# Define signal features (ChIP-seq histone marks and TFs)
signal_features = [
    "H3K27ac", "H3K4me3", "H3K27me3", "H3K9me3",
    "H3K4me1", "H3K36me3", "CTCF", "RAD21"
]

# Visualize raw distributions before normalization
plt.figure(figsize=(18, 10))
for i, feature in enumerate(signal_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(bins_df[feature], bins=100, kde=True)
    plt.title(f"Before normalization: {feature}")
plt.tight_layout()
plt.show()

# Normalize features:
# 1. Replace 0s with a small number for log transformation
# 2. Apply log1p transformation to compress dynamic range
# 3. Standardize using sklearn's StandardScaler
scaler = StandardScaler()
bins_df[signal_features] = bins_df[signal_features].replace(0, 1e-6)
bins_df[signal_features] = np.log1p(bins_df[signal_features])
bins_df[signal_features] = scaler.fit_transform(bins_df[signal_features])

# Visualize distributions after normalization
plt.figure(figsize=(18, 10))
for i, feature in enumerate(signal_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(bins_df[feature], bins=100, kde=True)
    plt.title(f"After normalization: {feature}")
plt.tight_layout()
plt.show()

# Label bins as accessible (1) or inaccessible (0) based on top 10% ATAC signal
threshold = np.percentile(bins_df["ATAC"], 90)
bins_df["label"] = (bins_df["ATAC"] >= threshold).astype(int)

# Save processed DataFrame for downstream training
bins_df.to_csv("data/labeled_colon_bins.csv", sep="\t", index=False)
