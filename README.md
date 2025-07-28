# Prediction of chromatin accessibility in colon cancer

## Project Overview:
This project applies a deep learning model to predict chromatin accessibility in the humon genome using ChIP-seq signals of histone markers and trascription factors, based on data from the HTC116 cell line.

## Rationale:
Understanding which regions of the genome are accessible is key to identifying active regulatory elements like promoters and enhancers. These open regions are where important processes—like gene transcription—begin. Instead of measuring accessibility directly, which can be time-consuming and costly, this project uses patterns in histone markers and transcription factors to predict whether a given region is accessible. By training a deep learning model on known ATAC-seq data, we can not only make accessibility predictions but also explore which epigenetic features contribute most to chromatin openness.

## Data used:
All datasets used in this project are publicly available from the ENCODE Project and originate from the HCT116 human colon cancer cell line:
- Accessibility reference:
  - ATAC-seq (ENCFF624HRW)
- Histone marks:
  - H3K27ac (ENCFF277XII)
  - H3K4me3 (ENCFF213WKK)
  - H3K27me3 (ENCFF457PEW)
  - H3K9me3 (ENCFF063OHO)
  - H3K4me1 (ENCFF182EJH)
  - H3K36me3 (ENCFF059WYR)
- Transcriprion factor:
  - CTCF (ENCFF813QCX)
  - RAD21 (ENCFF027QAE)

The bin coordinates:
- colon_1000bp_bins.bed –  is generated using hg38.chrom.sizes file from UCSC Genome Browser.

## Workflow Overview:
The project is structured into three main phases: data preparation, model training, and model evaluation.

### Phase 1: Data Preparation – `main_prepare.py`

- **Input**:
  - `colon_1000bp_bins.bed`: defines 1000bp bins across the genome
  - `.bigWig` signal files from ENCODE:
    - H3K27ac
    - H3K4me3
    - H3K27me3
    - H3K9me3
    - H3K4me1
    - H3K36me3
    - CTCF
    - RAD21
    - ATAC-seq (used for labeling)

- **Process**:
  - For each bin, compute the **mean signal** from each `.bigWig` file.
  - Applies **log1p + z-score normalization**
  - Labels bins:
    - `1` (accessible) if ATAC signal in top 10%  
    - `0` otherwise
  - Save results in a labeled CSV file.

- **Output**:
  - `labeled_colon_bins.csv`
 
### To get the data prepared:

**1. Clone repository**
```bash
git clone https://github.com/redaalomran/Colon_accessibility_prediction.git
cd Colon_accessibility_prediction
```

**2. Download .bigwig files**

Go to [ENCODE](https://www.encodeproject.org/) and download the `.bigWig` files listed in the **Data Used** section.

After downloading, place all `.bigWig` files in the `data/` directory.

Rename the files to match the filenames expected by the code.

| **Feature**    | **ENCODE File**              | **Rename to:**                         |
|----------------|------------------------------|----------------------------------------|
| **H3K27ac**    | `ENCFF277XII.bigWig`         | `ENCFF277XII_H3K27ac.bigWig`           |
| **H3K4me3**    | `ENCFF213WKK.bigWig`         | `ENCFF213WKK_H3K4me3.bigWig`           |
| **H3K27me3**   | `ENCFF457PEW.bigWig`         | `ENCFF457PEW_H3K27me3.bigWig`          |
| **H3K9me3**    | `ENCFF063OHO.bigWig`         | `ENCFF063OHO_H3K9me3.bigWig`           |
| **H3K4me1**    | `ENCFF182EJH.bigWig`         | `ENCFF182EJH_H3K4me1.bigWig`           |
| **H3K36me3**   | `ENCFF059WYR.bigWig`         | `ENCFF059WYR_H3K36me3.bigWig`          |
| **CTCF**       | `ENCFF813QCX.bigWig`         | `ENCFF813QCX_CTCF.bigWig`              |
| **RAD21**      | `ENCFF027QAE.bigWig`         | `ENCFF027QAE_RAD21.bigWig`             |
| **ATAC-seq**   | `ENCFF624HRW.bigWig`         | `ENCFF624HRW_ATAC.bigWig`              |

You can rename them manually or use the terminal commands below:

```bash
cd data
mv ENCFF277XII.bigWig ENCFF277XII_H3K27ac.bigWig
mv ENCFF213WKK.bigWig ENCFF213WKK_H3K4me3.bigWig
mv ENCFF457PEW.bigWig ENCFF457PEW_H3K27me3.bigWig
mv ENCFF063OHO.bigWig ENCFF063OHO_H3K9me3.bigWig
mv ENCFF182EJH.bigWig ENCFF182EJH_H3K4me1.bigWig
mv ENCFF059WYR.bigWig ENCFF059WYR_H3K36me3.bigWig
mv ENCFF813QCX.bigWig ENCFF813QCX_CTCF.bigWig
mv ENCFF027QAE.bigWig ENCFF027QAE_RAD21.bigWig
mv ENCFF624HRW.bigWig ENCFF624HRW_ATAC.bigWig
cd ..
```

**3. Download human genome sizes file**

```bash
curl -o data/hg38.chrom.sizes https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes
```

**4. Install requirement**

```bash
pip install -r requirements.txt
```

**5. Run the preparation code**

```bash
python scripts/main_prepare.py
```
---
### Phase 2: Model Training – `main_train.py`

- **Input**:
  - `labeled_colon_bins.csv`

- **Process**:
  - Split dataset for the Leave-One-Chromosome-Out cross validation:
    - `chr1` is used as the **test set**
    - All other chromosomes as the **training set**
  - Features:
    - H3K27ac, H3K4me3, H3K27me3, H3K9me3, H3K4me1, H3K36me3, CTCF, RAD21
  - Target:
    - ATAC-seq-based binary accessibility
  - Model:
    - Simple feedforward neural network
    - Binary Cross Entropy Loss with class balancing
  - Save model + test data + predictions

- **Output**:
  - `simple_nn_model.pth`
  - `X_test.npy`, `y_test.npy`
  - `probs.npy`, `preds.npy`, `labels.npy`

### To start the training process:
**Run the training code**
```bash
python scripts/main_train.py
```
---
### Phase 3.1: Model Evaluation – `main_evaluate.py`

- **Input**:
  - `probs.npy`, `preds.npy`, `labels.npy`

- **Metrics & Visualizations**:
  - Confusion Matrix
  - Classification Report
  - ROC Curve + AUC
  - Precision-Recall Curve
  - Histogram of predicted probabilities

### To evaluate the model:
**Run the evaluation code**
```bash
python scripts/main_evaluate.py
```

### Phase 3.2: Feature Importance – `feature_impact.py`

- **Input**:
  - `simple_nn_model.pth`, `X_test.npy`, `y_test.npy`

- **Process**:
  - Loads the trained neural network model
  - Defines a NumPy-based forward pass function for SHAP
  - Uses SHAP's `KernelExplainer` on a background of 100 random samples from `X_test`
  - Computes SHAP values on the first 100 test samples
  - Plots:
    - Global SHAP summary dot plot
	  - Global SHAP summary bar plot
	  - SHAP waterfall plot for a representative sample

- **Features analysed**:
  - H3K27ac, H3K4me3, H3K27me3, H3K9me3, H3K4me1, H3K36me3, CTCF, RAD21

- **Output**:
  - `shap_summary_plot.png` – Global SHAP dot plot (feature impact distribution)
  - `shap_summary_barplot.png` – SHAP bar plot (average absolute contribution)
  - `shap_waterfall_sample0.png` – Waterfall plot for the first test sample

### To run the SHAP-based feature interpretation:
**Run the feature interpreter code**
```bash
python scripts/main_feature_importance.py
```
