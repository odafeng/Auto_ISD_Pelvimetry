"""
Automated Pelvimetry and Body Composition Analysis Core Module
==============================================================
This module contains the core algorithms for the automated extraction of:
1. Inter-spinous Distance (ISD)
2. Anteroposterior Diameter (APD)
3. Posterior Pelvic Compartment Metrics (Triangle Area, Depth, Working Space)

The algorithms are designed to be robust against variations in patient positioning
and segmentation quality.

Dependencies: numpy, pandas, nibabel, scipy, skimage, matplotlib
Author: [Author Name]
Correspondence: [Email]
"""

import os
import time
import shlex
import subprocess
import traceback
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from skimage.draw import polygon
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

class PelvicConfig:
    """
    Configuration parameters for the pelvimetry pipeline.
    These parameters control the search windows, thresholds, and heuristic adjustments.
    """
    def __init__(self,
                 min_isd_mm=45.0,
                 max_isd_mm=130.0,
                 # Femur-based Search Window (relative to femoral head top)
                 search_margin_superior_mm=20.0,
                 search_margin_inferior_mm=70.0,
                 # Sacrum buffering for robustness
                 sacrum_buffer_slices=5,
                 # Edge margin to avoid selecting slices too close to scan boundaries
                 edge_margin_slices=2,
                 # Peak detection prominence (mm) for finding the "narrowest" point
                 prominence_mm=1.0,
                 # Smoothing window for the distance profile
                 smoothing_window=3,
                 # Plateau detection fallback
                 plateau_slope_thresh=1.0,   # mm/slice
                 plateau_min_len=3           # Minimum consecutive slices
                 ):
        self.min_isd_mm = min_isd_mm
        self.max_isd_mm = max_isd_mm
        self.search_margin_superior_mm = search_margin_superior_mm
        self.search_margin_inferior_mm = search_margin_inferior_mm
        self.sacrum_buffer_slices = sacrum_buffer_slices
        self.edge_margin_slices = edge_margin_slices
        self.prominence_mm = prominence_mm
        self.smoothing_window = smoothing_window
        self.plateau_slope_thresh = plateau_slope_thresh
        self.plateau_min_len = plateau_min_len

DEFAULT_CONFIG = PelvicConfig()

# =============================================================================
# Helper Functions
# =============================================================================

def _pick_spine_candidate_2d(pts_xy, sx, sy, midline_x, side="L",
                            posterior_band_q=0.20,
                            min_band_pts=80):
    """
    Identifies the candidate point for the ischial spine tip on a 2D slice.
    
    Logic:
    1. Gating: Restrict search to the posterior half of the hip bone to avoid anterior structures.
    2. Posterior Band: Select the most posterior quantile (e.g., bottom 20% in Y-axis) of points.
    3. Medial Selection: Within this band, select the point closest to the midline.
       - Tie-breaker: Choose the more posterior point.
    
    Args:
        pts_xy (np.array): (N, 2) array of pixel coordinates (x, y).
        sx, sy (float): Pixel spacing in mm.
        midline_x (float): X-coordinate of the pelvic midline.
        side (str): "L" or "R".
        posterior_band_q (float): Quantile threshold for the posterior band (0.0-1.0).
        min_band_pts (int): Minimum points required to form a reliable band.
        
    Returns:
        tuple: (candidate_point_xy, metadata_dict)
    """
    meta = {
        "n_total": 0,
        "side": side
    }

    if pts_xy is None or len(pts_xy) == 0:
        return None, meta

    meta["n_total"] = int(len(pts_xy))

    # 1. Posterior Gating (Y-axis centroid)
    # Assuming standard orientation where lower Y is posterior (or vice-versa, depending on coords).
    # Here we assume lower Y index is posterior? Let's check relative logic.
    # In DICOM/NIfTI, usually Y increases Anteriorly (RAS).
    # So Posterior is low Y.
    y_centroid = float(np.mean(pts_xy[:, 1]))
    
    # Filter for points more posterior (smaller Y) than the centroid
    posterior_pts = pts_xy[pts_xy[:, 1] < y_centroid]
    if len(posterior_pts) == 0:
        posterior_pts = pts_xy

    # 2. Posterior Band Selection
    # Select the quantile most posterior (Variance reduction)
    y_thr = np.quantile(posterior_pts[:, 1], posterior_band_q)
    band = posterior_pts[posterior_pts[:, 1] <= y_thr]

    # Fallback if band is too sparse (suggests fragmentation or thin slice)
    if len(band) < min_band_pts:
        band = posterior_pts

    # 3. Medial Selection Logic
    # Calculate distance to midline for all points in the band
    dx = np.abs(band[:, 0].astype(np.float32) - float(midline_x))
    y  = band[:, 1].astype(np.float32)

    # Primary sort: dx (ASC) - closest to midline
    # Secondary sort: y (ASC) - most posterior
    idx = np.lexsort((y, dx))
    
    return band[idx[0]], meta

# =============================================================================
# Main Analysis Class
# =============================================================================

class AutomatedPelvimetry:
    """
    Main controller for the pelvimetry analysis pipeline.
    """
    
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config

    def run_segmentation(self, input_nifti, output_dir, fast_mode=False):
        """
        Executes TotalSegmentator on the input NIfTI file.
        Requires 'TotalSegmentator' to be installed and accessible in the system PATH.
        """
        patient_id = os.path.basename(input_nifti).replace(".nii.gz", "").replace(".nii", "")
        patient_seg_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_seg_dir, exist_ok=True)

        rois = [
            'femur_left', 'femur_right',
            'hip_left', 'hip_right',
            'sacrum', 'colon'
        ]
        
        # Check if output already exists
        all_exist = all(
            os.path.exists(os.path.join(patient_seg_dir, f"{roi}.nii.gz"))
            for roi in rois
        )
        if all_exist:
            print(f"INFO: Segmentation for {patient_id} already exists. Skipping.")
            return patient_seg_dir

        print(f"PROCESSING: Running TotalSegmentator for {patient_id}...")
        start_time = time.time()

        try:
            # 1. Main Anatomy Segmentation
            cmd_main = [
                "TotalSegmentator", "-i", input_nifti, "-o", patient_seg_dir,
                "-rs", # Re-sample for speed (optional)
                "--roi_subset"
            ] + rois
            
            if fast_mode:
                cmd_main.append("--fast")

            # Execute command
            # Using shlex.quote for safety not strictly needed for list, but good practice if stringifying
            subprocess.run(cmd_main, check=True)

            # 2. Tissue Types (Fat/Muscle) - Requires separate task "tissue_types"
            # Note: This is computationally expensive.
            cmd_tissue = [
                "TotalSegmentator", "-i", input_nifti, "-o", patient_seg_dir,
                "-ta", "tissue_types"
            ]
            if fast_mode:
                cmd_tissue.append("--fast")
            
            subprocess.run(cmd_tissue, check=True)
            
            print(f"SUCCESS: Segmentation completed in {time.time() - start_time:.1f}s")
            return patient_seg_dir

        except subprocess.CalledProcessError as e:
            print(f"ERROR: Segmentation failed. {e}")
            return None

    def load_data(self, seg_dir):
        """
        Loads required NIfTI masks and ensures canonical RAS orientation.
        """
        data = {}
        files = {
            "hip_L": "hip_left.nii.gz",
            "hip_R": "hip_right.nii.gz",
            "sacrum": "sacrum.nii.gz",
            "femur_L": "femur_left.nii.gz",
            "femur_R": "femur_right.nii.gz",
            "colon": "colon.nii.gz",
            "torso_fat": "torso_fat.nii.gz"
        }
        
        affine = None
        spacing = (1.0, 1.0, 1.0) # sx, sy, sz

        for key, filename in files.items():
            path = os.path.join(seg_dir, filename)
            if os.path.exists(path):
                img = nib.load(path)
                img = nib.as_closest_canonical(img) # Standardization
                data[key] = img.get_fdata()
                
                if key == "hip_L":
                    affine = img.affine
                    spacing = img.header.get_zooms()
            else:
                data[key] = None

        return data, affine, spacing

    def calculate_isd(self, data, spacing):
        """
        Calculates the Inter-spinous Distance (ISD).
        
        Algorithm:
        1. Define Search Range: Based on Femoral Head location (+20mm / -70mm).
        2. Trace Profile: Calculate 'minimal inter-bone distance' for each slice in range.
        3. Valley Detection: Identify the anatomical characteristic of the ischial spines
           (protrusions narrowing the pelvic inlet) which correspond to local minima (valleys)
           in the distance profile.
        4. Plateau Fallback: If no distinct valley is found (e.g., due to flat anatomy),
           detect a stable 'plateau' of minimum distance.
           
        Returns:
            dict: Result containing ISD_mm, ISD_slice, and status.
        """
        res = {"Status": "Failed", "ISD_mm": None, "ISD_slice": None}
        sx, sy, sz = spacing
        
        # 1. Determine Search Range using Femoral Heads
        z_indices = []
        if data.get("femur_L") is not None: z_indices.append(np.where(data["femur_L"] > 0)[2])
        if data.get("femur_R") is not None: z_indices.append(np.where(data["femur_R"] > 0)[2])
        
        if not z_indices or len(np.concatenate(z_indices)) == 0:
            res["Status"] = "Failed_No_Femur"
            return res, []

        f_max = np.max(np.concatenate(z_indices)) # Top of femoral heads
        margin_sup = int(self.config.search_margin_superior_mm / sz)
        margin_inf = int(self.config.search_margin_inferior_mm / sz)
        
        z_start = max(0, f_max - margin_inf)
        z_end = min(data["hip_L"].shape[2], f_max + margin_sup)
        

        # 2. Scan Slices
        trace = []
        for z in range(z_start, z_end):
            rec = {
                "z": z,
                "dist_mm": np.nan,
                "pt_L": None,
                "pt_R": None,
                "midline_x": np.nan,
                "has_sacrum": False
            }

            # Check Sacrum
            if data["sacrum"] is not None:
                s_start = max(0, z - self.config.sacrum_buffer_slices)
                s_end = min(data["sacrum"].shape[2], z + self.config.sacrum_buffer_slices + 1)
                if np.sum(data["sacrum"][:, :, s_start:s_end]) > 0:
                    rec["has_sacrum"] = True
            
            # If no sacrum, we still append the record (with NaN) to maintain Z-continuity
            if not rec["has_sacrum"]:
                trace.append(rec)
                continue

            slice_L = data["hip_L"][:, :, z]
            slice_R = data["hip_R"][:, :, z]
            
            if np.sum(slice_L) == 0 or np.sum(slice_R) == 0:
                trace.append(rec)
                continue

            pts_L = np.argwhere(slice_L > 0)[:, :2]
            pts_R = np.argwhere(slice_R > 0)[:, :2]
            
            all_pts_x = np.concatenate([pts_L[:, 0], pts_R[:, 0]])
            midline_x = np.mean(all_pts_x)
            rec["midline_x"] = midline_x
            
            pt_L, _ = _pick_spine_candidate_2d(pts_L, sx, sy, midline_x, "L")
            pt_R, _ = _pick_spine_candidate_2d(pts_R, sx, sy, midline_x, "R")
            
            if pt_L is not None and pt_R is not None:
                rec["dist_mm"] = np.linalg.norm((pt_L - pt_R) * np.array([sx, sy]))
                rec["pt_L"] = pt_L
                rec["pt_R"] = pt_R
            
            trace.append(rec)

        df_trace = pd.DataFrame(trace)
        if df_trace.empty or df_trace['dist_mm'].isna().all():
            res["Status"] = "Failed_No_Valid_Slices"
            return res, trace

        # 3. Curve Analysis
        dists = df_trace['dist_mm'].interpolate(limit_direction='both').to_numpy()
        kernel = np.ones(self.config.smoothing_window) / self.config.smoothing_window
        dists_smooth = np.convolve(dists, kernel, mode='same')
        
        # Update trace with smoothed values for QC plotting
        df_trace['smooth_dist'] = dists_smooth
        trace = df_trace.to_dict('records') # Update the returned trace list
        
        # Find Valleys (inverted peaks)
        peaks, properties = find_peaks(-dists_smooth, prominence=self.config.prominence_mm)
        
        candidates = []
        for idx in peaks:
            z_val = int(df_trace.iloc[idx]['z'])
            dist_val = df_trace.iloc[idx]['dist_mm']
            prominence = properties['prominences'][list(peaks).index(idx)]
            has_sacrum = df_trace.iloc[idx]['has_sacrum']

            # Edge Sanity Check
            if (z_val <= z_start + self.config.edge_margin_slices) or \
               (z_val >= z_end - self.config.edge_margin_slices):
               continue

            # Sacrum Sanity Check (Valley must have sacrum)
            if not has_sacrum:
                continue

            # Range sanity check
            if self.config.min_isd_mm <= dist_val <= self.config.max_isd_mm:
                candidates.append((z_val, dist_val, prominence))
        
        best_rec = None
        
        # Strategy A: Distinct Valley
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best_z = candidates[0][0]
            # Retrieve from df_trace where z matches
            best_rec = df_trace[df_trace['z'] == best_z].iloc[0].to_dict()
            res["Status"] = "Success_Valley"

        # Strategy B: Plateau Fallback
        elif len(df_trace) > 5:
            slope = np.gradient(dists_smooth)
            is_plateau = np.abs(slope) < self.config.plateau_slope_thresh
            
            df_trace['is_plateau'] = is_plateau
            # Group consecutive True values for plateau
            df_trace['group'] = (df_trace['is_plateau'] != df_trace['is_plateau'].shift()).cumsum()
            
            valid_plateaus = []
            for _, grp in df_trace[df_trace['is_plateau']].groupby('group'):
                if len(grp) < self.config.plateau_min_len:
                    continue
                
                # Representative slice (median of the plateau group)
                median_idx = len(grp) // 2
                rep_row = grp.iloc[median_idx]
                z_rep = int(rep_row['z'])
                dist_rep = rep_row['dist_mm']
                
                # Sanity checks
                if (z_rep <= z_start + self.config.edge_margin_slices) or \
                   (z_rep >= z_end - self.config.edge_margin_slices):
                    continue
                
                if not rep_row['has_sacrum']:
                    continue
                    
                if self.config.min_isd_mm <= dist_rep <= self.config.max_isd_mm:
                    valid_plateaus.append({
                        'z': z_rep,
                        'dist_mm': dist_rep,
                        'row_idx': rep_row.name,
                        'plateau_len': len(grp)
                    })
            
            if valid_plateaus:
                # Pick the narrowest plateau
                best_plateau = min(valid_plateaus, key=lambda x: x['dist_mm'])
                best_rec = df_trace.loc[best_plateau['row_idx']].to_dict()
                res["Status"] = "Success_Plateau"


        if best_rec:
            res["ISD_mm"] = best_rec["dist_mm"]
            res["ISD_slice"] = int(best_rec["z"])
            res["pt_L"] = best_rec["pt_L"] # (x, y)
            res["pt_R"] = best_rec["pt_R"] # (x, y)
            res["midline_x"] = best_rec["midline_x"]
            
        # Debug Info for QC Plotting
        res["debug_valleys"] = valleys if 'valleys' in locals() else []
        res["debug_search_range"] = (z_min_search, z_max_search) if 'z_min_search' in locals() else None
        
        return res, trace

    def calculate_apd(self, data, isd_res, spacing):
        """
        Calculates the Anteroposterior Diameter (APD) at the ISD level.
        Defined as the distance between the posterior-most aspect of the pubic symphysis
        and the anterior-most aspect of the sacrum (Point C).
        """
        res = {"APD_mm": None, "APD_Status": "Failed"}
        
        if not isd_res["Status"].startswith("Success"):
            res["APD_Status"] = "Failed_No_ISD"
            return res

        z = isd_res["ISD_slice"]
        sx, sy, sz = spacing
        midline_x = isd_res["midline_x"]

        # Masks
        if data["hip_L"] is None or data["hip_R"] is None or data["sacrum"] is None:
             res["APD_Status"] = "Failed_Missing_Masks"
             return res

        slice_hips = (data["hip_L"][:,:,z] > 0) | (data["hip_R"][:,:,z] > 0)
        slice_sacrum = data["sacrum"][:,:,z] > 0

        pts_hips = np.argwhere(slice_hips)
        pts_sacrum = np.argwhere(slice_sacrum)

        if len(pts_hips) == 0 or len(pts_sacrum) == 0:
            res["APD_Status"] = "Failed_Empty_Masks"
            return res

        # 1. Identify Pubis Point (pt_P)
        # Strategy: Anterior-most part of the hip mask near midline.
        # In RAS (assuming Y increases Anteriorly), this is Max Y.
        y_max_hip = np.max(pts_hips[:, 1])
        # Filter points within 10mm of the very front
        candidates_P = pts_hips[pts_hips[:, 1] > (y_max_hip - 10/sy)]
        
        if len(candidates_P) == 0:
             res["APD_Status"] = "Failed_No_Pubis_Candidate"
             return res
             
        # Pick point closest to midline
        pt_P = candidates_P[np.argmin(np.abs(candidates_P[:, 0] - midline_x))]

        # 2. Identify Sacrum Point (pt_S) - Same as Triangle Point C
        # Filter for points roughly central (near midline)
        sacrum_mid = pts_sacrum[np.abs(pts_sacrum[:, 0] - midline_x) < (30/sx)]
        
        if len(sacrum_mid) == 0:
            res["APD_Status"] = "Failed_No_Sacrum_Midline"
            return res
            
        # Anterior-most point of sacrum (Max Y)
        pt_S = sacrum_mid[np.argmax(sacrum_mid[:, 1])]

        # 3. Calculate Distance
        apd_mm = abs(pt_P[1] - pt_S[1]) * sy
        res["APD_mm"] = round(apd_mm, 2)
        res["APD_Status"] = "Success"
        res["pt_P"] = pt_P
        res["pt_S"] = pt_S
        
        return res

    def calculate_triangle_metrics(self, data, isd_res, spacing):
        """
        Calculates measures of the posterior pelvic compartment defined by the triangle
        formed by the two ischial spines and the anterior sacrum.
        """
        res = {
            "Triangle_Area_cm2": None,
            "Triangle_Depth_mm": None, # Triangle height
            "Triangle_Shape_Index": None,
            "Bowel_Area_Triangle_cm2": None,
            "Bowel_Occupancy_Ratio": None,
            "pPFA_cm2": None,          # Posterior Pelvic Fat Area (formerly Fat_Area_cm2)
            "Fat_Occupancy_Ratio": None,
            "Working_Space_cm2": None
        }
        
        if not isd_res["Status"].startswith("Success"):
            return res

        z = isd_res["ISD_slice"]
        pt_A = isd_res["pt_L"] # Left Spine
        pt_B = isd_res["pt_R"] # Right Spine
        midline_x = isd_res["midline_x"]
        
        sx, sy, sz = spacing
        
        # 1. Define Point C (Posterior Vertex)
        # Point C is defined as the most anterior point of the sacrum at the ISD level.
        if data["sacrum"] is None or np.sum(data["sacrum"][:,:,z]) == 0:
            return res # Cannot define triangle without sacrum
            
        pts_sac = np.argwhere(data["sacrum"][:,:,z] > 0)
        # Filter for points roughly central (near midline) to avoid lateral sacral ala
        valid_sac = pts_sac[np.abs(pts_sac[:, 0] - midline_x) < (40 / sx)]
        
        if len(valid_sac) == 0:
            return res

        # RAS coordinate system: Y axis increases anteriorly.
        # So the "Anterior Sacral Cortex" is the point with MAX Y in the sacrum mask.
        pt_C = valid_sac[np.argmax(valid_sac[:, 1])]
        res["Point_C_x"] = int(pt_C[0])
        res["Point_C_y"] = int(pt_C[1])
        
        # 2. Geometric Calculation (Shoelace Formula)
        # Convert to physical units (mm)
        Ax, Ay = pt_A[0]*sx, pt_A[1]*sy
        Bx, By = pt_B[0]*sx, pt_B[1]*sy
        Cx, Cy = pt_C[0]*sx, pt_C[1]*sy
        
        area_mm2 = 0.5 * abs(Ax*(By - Cy) + Bx*(Cy - Ay) + Cx*(Ay - By))
        res["Triangle_Area_cm2"] = round(area_mm2 / 100, 2)
        
        # Depth (Height of triangle from base AB to vertex C)
        base_len = isd_res["ISD_mm"]
        if base_len > 0:
            res["Triangle_Depth_mm"] = round((2 * area_mm2) / base_len, 2)
            res["Triangle_Shape_Index"] = round(res["Triangle_Depth_mm"] / base_len, 3)
            
        # 3. Content Analysis within Triangle
        # We create a boolean mask of the triangle to check overlap with organs
        shape = data["hip_L"].shape[:2]
        r = np.array([pt_A[0], pt_B[0], pt_C[0]])
        c = np.array([pt_A[1], pt_B[1], pt_C[1]])
        rr, cc = polygon(r, c, shape)
        mask_tri = np.zeros(shape, dtype=bool)
        mask_tri[rr, cc] = True
        
        pixel_area_cm2 = (sx * sy) / 100.0
        triangle_area_cm2_val = area_mm2 / 100.0
        
        # bowel Intersection
        if data["colon"] is not None:
            mask_bowel = data["colon"][:,:,z] > 0
            bowel_area = np.sum(mask_tri & mask_bowel) * pixel_area_cm2
            res["Bowel_Area_Triangle_cm2"] = round(bowel_area, 2)
            
            if triangle_area_cm2_val > 0:
                res["Bowel_Occupancy_Ratio"] = round(bowel_area / triangle_area_cm2_val, 3)
        else:
            res["Bowel_Area_Triangle_cm2"] = 0.0
            
        # Fat Intersection -> pPFA
        if data["torso_fat"] is not None:
            mask_fat = data["torso_fat"][:,:,z] > 0
            fat_area = np.sum(mask_tri & mask_fat) * pixel_area_cm2
            res["pPFA_cm2"] = round(fat_area, 2)
            
            if triangle_area_cm2_val > 0:
                res["Fat_Occupancy_Ratio"] = round(fat_area / triangle_area_cm2_val, 3)
        else:
            res["pPFA_cm2"] = 0.0
            
        # Working Space
        res["Working_Space_cm2"] = round(
            max(0, res["Triangle_Area_cm2"] - res["Bowel_Area_Triangle_cm2"] - res["pPFA_cm2"]), 2
        )
        
        return res

    def pipeline_single_case(self, nifti_path, seg_output_dir):
        """
        Runs the full pipeline for a single NIfTI file.
        """
        results = {"Patient_ID": os.path.basename(nifti_path)}
        
        # 1. Segmentation (External Tool)
        # Assumes segmentation is already run or runs it now
        seg_dir = self.run_segmentation(nifti_path, seg_output_dir)
        if not seg_dir:
            results["Status"] = "Segmentation_Failed"
            return results
        
        # 2. Load Masks
        data, affine, spacing = self.load_data(seg_dir)
        
        # 3. ISD
        isd_res, trace = self.calculate_isd(data, spacing)
        results.update(isd_res)
        
        # 4. APD
        apd_res = self.calculate_apd(data, isd_res, spacing)
        results.update(apd_res)

        # 5. Triangle Metrics
        tri_res = self.calculate_triangle_metrics(data, isd_res, spacing)
        results.update(tri_res)
        
        # Flatten Coordinates for Easy Reporting
        if "pt_L" in isd_res and isd_res["pt_L"] is not None:
            results["ISD_L_x"] = int(isd_res["pt_L"][0])
            results["ISD_L_y"] = int(isd_res["pt_L"][1])
        else:
            results["ISD_L_x"] = None; results["ISD_L_y"] = None
            
        if "pt_R" in isd_res and isd_res["pt_R"] is not None:
            results["ISD_R_x"] = int(isd_res["pt_R"][0])
            results["ISD_R_y"] = int(isd_res["pt_R"][1])
        else:
            results["ISD_R_x"] = None; results["ISD_R_y"] = None

        if "Point_C_x" in tri_res:
            results["Point_C_x"] = tri_res["Point_C_x"]
            results["Point_C_y"] = tri_res["Point_C_y"]
        else:
            results["Point_C_x"] = None; results["Point_C_y"] = None
            
        # Clean up nested tuples if desired, but keeping them for programmatic use is fine.
        
        return results

if __name__ == "__main__":
    # Internal Unit Test
    pass
