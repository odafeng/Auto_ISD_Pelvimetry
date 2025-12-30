#!/usr/bin/env python
# coding: utf-8

# # QC Figure Generation Demo
# 
# This notebook demonstrates how to generate the quality control (QC) figure for publication, containing:
# (a) Anatomy overlay with ISD landmarks
# (b) ISD Search Profile with valleys and smoothing trace
# 
# It uses the `Demo` data and `pelvimetry_core`.

# In[ ]:


import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pelvimetry_core import AutomatedPelvimetry, PelvicConfig

# Ensure plots display inline (Notebook only)
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Load Data & Run Pipeline

# In[ ]:


pipeline = AutomatedPelvimetry()
demo_dir = "./Demo"
nifti_file = os.path.join(demo_dir, "Patient_CT.nii.gz")

# Load Masks
print("Loading data...")
data, affine, spacing = pipeline.load_data(demo_dir)

# Run ISD Calculation
print("Calculating ISD...")
isd_res, trace = pipeline.calculate_isd(data, spacing)


# ## 2. Load CT Slice Image
# We need the actual CT pixel data for the background anatomy.

# In[ ]:


z_slice = isd_res.get("ISD_slice")
if z_slice is not None:
    print(f"Loading Slice Z={z_slice}...")
    ct_img = nib.load(nifti_file)
    ct_img = nib.as_closest_canonical(ct_img)
    ct_data = ct_img.get_fdata()
    ct_slice = ct_data[:, :, z_slice]
else:
    print("Error: No ISD slice found.")
    ct_slice = None


# ## 3. Define Plotting Function
# This is adapted from Cell 10 of the main pipeline notebook.

# In[ ]:


def save_publication_figure(patient_id, ct_slice, res, trace, config, sx, sy):
    """
    Generate QC figure with (a) Anatomy and (b) Search Profile.
    """
    # Extract debug info from res
    valleys = res.get("debug_valleys", [])
    search_range = res.get("debug_search_range", None)
    status = res.get("Status", "Failed")

    fig = plt.figure(figsize=(12, 6), dpi=100) # dpi=100 for screen, 600 for file
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.25)

    # Main Title
    fig.suptitle(f"Patient ID: {patient_id}", fontsize=16, fontweight='bold', y=0.98)

    # --- Panel A: Anatomy Overlay ---
    ax1 = fig.add_subplot(gs[0])

    if ct_slice is not None:
        # Bone Window (-150 ~ 1000)
        ax1.imshow(ct_slice.T, cmap='gray', origin="lower", aspect=sy/sx, vmin=-150, vmax=1000)

    is_success = status.startswith("Success")

    if is_success:
        midline_x = res.get("midline_x")
        # The pipeline flattens pts to lists/tuples, pelvimetry_core output keeps tuples in pt_L/R
        # But pipeline_single_case output flattens them. Here we use `isd_res` from core directly.
        pt_L = res.get("pt_L") 
        pt_R = res.get("pt_R")
        isd_slice = res.get("ISD_slice")
        isd_mm = res.get("ISD_mm")

        # Draw Geometry
        if midline_x: 
            ax1.axvline(x=midline_x, color='yellow', linestyle="--", alpha=0.6, linewidth=1)

        if pt_L is not None: 
            # Note: Core pt_L is (x, y) tuple/array
            ax1.scatter([pt_L[0]], [pt_L[1]], c='red', s=80, marker="x", linewidth=2.5, zorder=5)
        if pt_R is not None: 
            ax1.scatter([pt_R[0]], [pt_R[1]], c='blue', s=80, marker="x", linewidth=2.5, zorder=5)

        if pt_L is not None and pt_R is not None:
            ax1.plot([pt_L[0], pt_R[0]], [pt_L[1], pt_R[1]], c='cyan', linewidth=2.5, linestyle='-', alpha=0.9, zorder=4)

        title_text = f"ISD: {isd_mm:.1f} mm\n(Slice Z={isd_slice})"
    else:
        title_text = f"FAILED\n{status}"

    ax1.set_title(f"(a) Anatomy (Z={res.get('ISD_slice', '?')})", loc='left', fontsize=12, fontweight='bold')
    ax1.text(0.05, 0.95, title_text, transform=ax1.transAxes, color='white' if is_success else 'red',
             verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round", fc="black", ec="none", alpha=0.5))
    ax1.axis('off')

    # --- Panel B: Search Profile ---
    ax2 = fig.add_subplot(gs[1])

    df = pd.DataFrame(trace)
    if 'dist_mm' in df.columns and not df['dist_mm'].dropna().empty:
        valid_df = df.dropna(subset=['dist_mm'])

        # 1. Search Range
        if search_range:
            z_s, z_e = search_range
            ax2.axvspan(z_s, z_e, color='green', alpha=0.1, label='Search Window')

        # 2. Raw & Smooth Data
        ax2.plot(valid_df['z'], valid_df['dist_mm'], marker='o', markersize=2, linestyle='-', color='silver', alpha=0.6, label='Raw Dist')
        if 'smooth_dist' in valid_df.columns:
            ax2.plot(valid_df['z'], valid_df['smooth_dist'], color='#1f77b4', linewidth=2, label='Smoothed')

        # 3. Threshold Line
        ax2.axhline(y=config.min_isd_mm, color='gray', linestyle='--', linewidth=1, label='Min Thresh')

        # 4. Valleys
        if valleys:
            for v_z, v_dist, v_prom in valleys:
                ax2.scatter([v_z], [v_dist], color='orange', s=50, marker='v', zorder=4, alpha=0.8)

        # 5. Selected Point
        if is_success:
            ax2.scatter([res["ISD_slice"]], [res["ISD_mm"]], color='red', s=120, marker='*', zorder=6, label='Selected Spine', edgecolor='black')
            ax2.annotate(f"{res['ISD_mm']:.1f}", (res["ISD_slice"], res["ISD_mm"]),
                         xytext=(0, 10), textcoords='offset points', ha='center', color='red', fontweight='bold')

    ax2.set_title("(b) ISD Search Profile", loc='left', fontsize=12, fontweight='bold')
    ax2.set_xlabel("Z-Slice Index (Inferior $\\to$ Superior)", fontsize=10)
    ax2.set_ylabel("Inter-spinous Distance (mm)", fontsize=10)
    ax2.legend(fontsize=8, loc='upper right', frameon=True)

    plt.tight_layout()
    return fig


# ## 4. Generate Plot

# In[ ]:


config = PelvicConfig() # Default config for plotting threshold lines
fig = save_publication_figure("Demo_Patient", ct_slice, isd_res, trace, config, spacing[0], spacing[1])

# Save to file
fig.savefig("Demo_QC_Plot.png", dpi=300)
print("QC plot saved to Demo_QC_Plot.png")

