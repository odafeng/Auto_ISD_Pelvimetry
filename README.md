# Automated Pelvimetry and Body Composition Analysis

This repository contains the source code for the automated pelvimetry pipeline described in the manuscript.

## Overview

The pipeline automates the extraction of key anatomical metrics from CT scans:
1.  **Inter-spinous Distance (ISD)**: Minimum distance between ischial spines using valley detection or plateau fallback.
2.  **Mid-pelvic Anteroposterior Diameter (mAPD)**: Distance between the pubic symphysis and sacral promontory equivalent.
3.  **Pelvic Triangle Metrics**: Area, Depth, Shape Index, **pPFA** (Posterior Pelvic Fat Area), **Working Space**, and Occupancy Ratios.

## Prerequisites

1.  **Python 3.8+**
2.  **Input Data Requirements**: 
    To run the analysis code, you must first process your CT scans to obtain:
    *   **Original CT Image**: In NIfTI format (`.nii.gz`).
    *   **Segmentation Masks**: Generated via [TotalSegmentator](https://github.com/wasserth/TotalSegmentator). The following masks are required:
        *   `femur_left.nii.gz`, `femur_right.nii.gz`
        *   `hip_left.nii.gz`, `hip_right.nii.gz`
        *   `sacrum.nii.gz`
        *   `colon.nii.gz`
        *   `torso_fat.nii.gz` (for body composition)

    > **Citation for TotalSegmentator**:
    > Wasserthal, J., Breit, H. C., Meyer, M. T., Pradella, M., Hinck, D., Sauter, A. W., Heye, T., Boll, D. T., Cyriac, J., Yang, S., Bach, M., & Segeroth, M. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology. Artificial intelligence, 5(5), e230024. [https://doi.org/10.1148/ryai.230024]

## Installation

1.  Clone this repository or download the source code.
2.  Install Python dependencies:

```bash
pip install -r requirements.txt
```

3.  Ensure `TotalSegmentator` is installed if you need to generate masks from scratch.

## Usage

### 1. Demo Data
A `Demo` folder is included in this package containing a sample patient:
*   `Demo/Patient_CT.nii.gz`: Original CT.
*   `Demo/*.nii.gz`: Pre-computed segmentation masks.
You can use this data to verify the code immediately.

### 2. running the Code
The provided code performs two distinct tasks:

#### A. Metric Calculation (`Pelvimetry_Demo.ipynb`)
Calculates all anatomical metrics (ISD, APD, Triangle Area, pPFA, Working Space, etc.) and outputs them numerically.
*   **Input**: NIfTI file + Segmentation Masks folder.
*   **Output**: Printed metrics and coordinate points.

#### B. QC Figure Generation (`Generate_QC_Plot.ipynb`)
Generates a publication-quality Quality Control (QC) figure showing the anatomical landmarks and the ISD search profile curve.
*   **Input**: NIfTI file + Segmentation Masks folder.
*   **Output**: A high-resolution PNG file (`Demo_QC_Plot.png`) visualizing the analysis.

### 3. Library Usage
The core logic is encapsulated in `pelvimetry_core.py`. You can import `AutomatedPelvimetry` to analyze your own NIfTI files.

```python
from pelvimetry_core import AutomatedPelvimetry

# Initialize
pipeline = AutomatedPelvimetry()

# Run Analysis
# Note: Ensure segmentation masks are present in 'seg_output_dir'
results = pipeline.pipeline_single_case(
    nifti_path="path/to/patient_ct.nii.gz",
    seg_output_dir="path/to/output_folder_with_masks"
)

print(f"ISD: {results['ISD_mm']} mm")
```

## License
[MIT License / Academic Use Only]
