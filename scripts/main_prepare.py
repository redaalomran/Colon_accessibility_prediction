import numpy as np
import pandas as pd
import pyBigWig
import os

def generate_bins(chrom_sizes_path, output_path, bin_size=1000):
    with open(chrom_sizes_path) as f, open(output_path, "w") as out:
        for line in f:
            chrom, size = line.strip().split("\t")
            size = int(size)
            for start in range(0, size, bin_size):
                end = min(start + bin_size, size)
                out.write(f"{chrom}\t{start}\t{end}\n")

if os.path.exists("data/colon_1000bp_bins.bed"):
    print("colon_1000bp_bins.bed already exists. Skipping generation.")
else:
    print("Generating 1000bp bins from hg38.chrom.sizes...")
    generate_bins("data/hg38.chrom.sizes", "data/colon_1000bp_bins.bed")
    print("Generation complete: colon_1000bp_bins.bed")

bins_df = pd.read_csv("data/colon_1000bp_bins.bed", sep="\t", names=["chrom", "start", "end"])
print("Total bins:", len(bins_df))

bw_H3K27ac = pyBigWig.open("data/ENCFF277XII_H3K27ac.bigWig")
bw_H3K4me3 = pyBigWig.open("data/ENCFF213WKK_H3K4me3.bigWig")
bw_H3K27me3 = pyBigWig.open("data/ENCFF457PEW_H3K27me3.bigWig")
bw_CTCF = pyBigWig.open("data/ENCFF813QCX_CTCF.bigWig")
bw_ATAC = pyBigWig.open("data/ENCFF624HRW_ATAC.bigWig")

h3k27ac_vals = []
h3k4me3_vals = []
h3k27me3_vals = []
ctcf_vals = []
atac_vals = []

def safe_mean_signal(bw, chrom, start, end):
    try:
        values = np.array(bw.values(chrom, start, end, numpy=True))
        values = values[~np.isnan(values)]
        return np.mean(values) if len(values) > 0 else 0.0
    except:
        return 0.0

from tqdm import tqdm

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
            ctcf_vals.append(safe_mean_signal(bw_CTCF, chrom, start, end))
        except:
            ctcf_vals.append(0.0)

        try:
            atac_vals.append(safe_mean_signal(bw_ATAC, chrom, start, end))
        except:
            atac_vals.append(0.0)
    else:
        h3k27ac_vals.append(0.0)
        h3k4me3_vals.append(0.0)
        h3k27me3_vals.append(0.0)
        ctcf_vals.append(0.0)
        atac_vals.append(0.0)

bins_df["H3K27ac"] = h3k27ac_vals
bins_df["H3K4me3"] = h3k4me3_vals
bins_df["H3K27me3"] = h3k27me3_vals
bins_df["CTCF"] = ctcf_vals
bins_df["ATAC"] = atac_vals
bins_df["label"] = (bins_df["ATAC"] > 0.5).astype(int)
bins_df.to_csv("data/lableled_colon_bins.csv", sep="\t", index=False)
