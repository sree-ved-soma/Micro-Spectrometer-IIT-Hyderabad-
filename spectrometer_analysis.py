"""
Micro-Spectrometer — Spectral Analysis Script
IIT Hyderabad | Soma Sree Ved

Description:
    Loads a raw spectrum image captured through the micro-spectrometer,
    extracts the spectral band, calibrates pixel positions to wavelengths
    (380–750 nm), and plots:
        1. Total intensity vs wavelength
        2. Reconstructed colour bar from actual pixel data
        3. RGB channel breakdown vs wavelength

Usage:
    Run the script and enter the full path to your spectrum image when prompted.
    Output plot is saved automatically as <image_name>_spectrum_plot.png
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import subprocess
import sys


# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION CONSTANTS
# Based on white light image analysis of the left-side diffraction pattern.
# Pixel positions are stored as relative fractions of image width so that
# the calibration works regardless of image resolution.
#
#   Relative position 0.108  →  700 nm  (red end)
#   Relative position 0.247  →  450 nm  (blue end)
#
# Note: On the left side of the diffraction pattern, pixel index increases
# as wavelength DECREASES (red → blue), so the slope is negative.
# ──────────────────────────────────────────────────────────────────────────────

REL_PX_REDEND  = 0.108   # ~11% across image width = red end  (700 nm)
REL_PX_BLUEEND = 0.247   # ~25% across image width = blue end (450 nm)
WL_REDEND      = 700.0   # nm
WL_BLUEEND     = 450.0   # nm

BAND_HALF = 10   # pixels above and below the peak row to average over
SMOOTH    = 5    # moving average window size for noise reduction


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Moving average smoother
# ──────────────────────────────────────────────────────────────────────────────

def smooth(arr, window):
    """Apply a simple moving average to reduce noise in spectral data."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Load image
# ──────────────────────────────────────────────────────────────────────────────

image_path = input("Enter full path to your spectrum image: ").strip()

# Strip surrounding quotes if user dragged the file in
while image_path and image_path[0] in ('"', "'") and image_path[0] == image_path[-1]:
    image_path = image_path[1:-1].strip()

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

img  = Image.open(image_path).convert("RGB")
data = np.array(img)
H, W = data.shape[:2]
print(f"\nImage loaded: {W} x {H} pixels")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Scale calibration to actual image width
# ──────────────────────────────────────────────────────────────────────────────

px_red  = int(REL_PX_REDEND  * W)
px_blue = int(REL_PX_BLUEEND * W)
print(f"Calibration: pixel {px_red} = {WL_REDEND} nm  |  pixel {px_blue} = {WL_BLUEEND} nm")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Extract the spectral band
# Find the brightest row (peak of the spectral line) and average BAND_HALF
# rows above and below it to reduce sensor noise.
# ──────────────────────────────────────────────────────────────────────────────

peak_row = int(np.argmax(data.mean(axis=(1, 2))))
r0 = max(0, peak_row - BAND_HALF)
r1 = min(H, peak_row + BAND_HALF + 1)

band = data[r0:r1, :, :].astype(float)

R     = smooth(band[:, :, 0].mean(axis=0), SMOOTH)
G     = smooth(band[:, :, 1].mean(axis=0), SMOOTH)
B     = smooth(band[:, :, 2].mean(axis=0), SMOOTH)
total = (R + G + B) / 3.0

pixels = np.arange(W)
print(f"Spectral band: rows {r0} – {r1}  (peak row = {peak_row})")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Convert pixel positions to wavelengths (linear calibration)
# slope is negative because wavelength decreases as pixel index increases
# on the left side of the diffraction pattern.
# ──────────────────────────────────────────────────────────────────────────────

slope      = (WL_REDEND - WL_BLUEEND) / (px_red - px_blue)
intercept  = WL_BLUEEND - slope * px_blue
wavelength = slope * pixels + intercept

# Keep only the visible range (380–750 nm) from the left half of the image
vis = (wavelength >= 380) & (wavelength <= 750) & (pixels <= W // 2)
if vis.sum() < 10:
    # Fallback: use full width if left-half filter is too aggressive
    vis = (wavelength >= 380) & (wavelength <= 750)

wl  = wavelength[vis]
R_v = R[vis]
G_v = G[vis]
B_v = B[vis]
T_v = total[vis]
print(f"Wavelength range: {wl.min():.1f} – {wl.max():.1f} nm  ({vis.sum()} pixels used)")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Sort by wavelength so the colour bar runs violet → red left to right
# ──────────────────────────────────────────────────────────────────────────────

sort_idx  = np.argsort(wl)
wl_sorted = wl[sort_idx]
R_s = R_v[sort_idx]
G_s = G_v[sort_idx]
B_s = B_v[sort_idx]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: Plot — three-panel figure
#   Panel 1 (top)    : Total intensity vs wavelength with colour landmark lines
#   Panel 2 (middle) : Reconstructed colour bar from actual pixel RGB data
#   Panel 3 (bottom) : Individual R, G, B channel intensities vs wavelength
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    3, 1,
    figsize=(14, 10),
    gridspec_kw={"height_ratios": [3, 1, 2]}
)
fig.patch.set_facecolor("#0d0d0d")
for ax in axes:
    ax.set_facecolor("#0d0d0d")

ax0, ax1, ax2 = axes

# --- Panel 1: Total intensity ---
ax0.fill_between(wl, T_v, alpha=0.25, color="white")
ax0.plot(wl, T_v, color="white", linewidth=1.4, label="Total intensity")
ax0.set_ylabel("Mean intensity (0–255)", color="white", fontsize=11)
ax0.set_title(
    "Spectrum — Intensity vs Wavelength  λ (nm)",
    color="white", fontsize=14, fontweight="bold", pad=12
)
ax0.tick_params(colors="white")
ax0.spines[:].set_color("#444")
ax0.set_xlim(380, 750)
ax0.set_ylim(0, 270)
ax0.legend(facecolor="#1a1a1a", edgecolor="#555", labelcolor="white")
ax0.grid(axis="y", color="#333", linewidth=0.5)

# Colour landmark vertical lines
landmarks = [
    (420, "Violet", "#9400D3"),
    (470, "Blue",   "#0000FF"),
    (520, "Green",  "#00AA00"),
    (580, "Yellow", "#CCCC00"),
    (620, "Orange", "#FF7700"),
    (680, "Red",    "#FF0000"),
]
for wl_c, label, col in landmarks:
    ax0.axvline(wl_c, color=col, alpha=0.35, linewidth=0.8, linestyle="--")
    ax0.text(wl_c, 255, label, color=col, fontsize=7,
             ha="center", va="bottom", alpha=0.9)

# --- Panel 2: Colour bar from actual pixel data ---
bar = np.stack([R_s, G_s, B_s], axis=1).astype(np.uint8)[np.newaxis, :, :]
ax1.imshow(
    np.clip(bar, 0, 255),
    aspect="auto",
    extent=[wl_sorted[0], wl_sorted[-1], 0, 1]
)
ax1.set_yticks([])
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#444")
ax1.set_xlim(380, 750)
ax1.set_ylabel("Spectrum", color="white", fontsize=10)

# --- Panel 3: RGB channels ---
ax2.fill_between(wl_sorted, R_s, alpha=0.20, color="red")
ax2.fill_between(wl_sorted, G_s, alpha=0.20, color="lime")
ax2.fill_between(wl_sorted, B_s, alpha=0.20, color="dodgerblue")
ax2.plot(wl_sorted, R_s, color="red",        linewidth=1.0, label="Red channel")
ax2.plot(wl_sorted, G_s, color="lime",       linewidth=1.0, label="Green channel")
ax2.plot(wl_sorted, B_s, color="dodgerblue", linewidth=1.0, label="Blue channel")
ax2.set_xlabel("Wavelength  λ (nm)", color="white", fontsize=12)
ax2.set_ylabel("Channel intensity",  color="white", fontsize=11)
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#444")
ax2.set_xlim(380, 750)
ax2.set_ylim(0, 270)
ax2.legend(facecolor="#1a1a1a", edgecolor="#555", labelcolor="white", ncol=3)
ax2.grid(axis="y", color="#333", linewidth=0.5)

# Shared x-axis ticks for all panels
ticks = [400, 450, 500, 550, 600, 650, 700, 750]
for ax in [ax0, ax1, ax2]:
    ax.set_xticks(ticks)

plt.tight_layout(pad=1.5)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7: Save and open output
# ──────────────────────────────────────────────────────────────────────────────

out_path = os.path.splitext(image_path)[0] + "_spectrum_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nPlot saved → {out_path}")

try:
    if sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", out_path])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", out_path])
    elif sys.platform == "win32":
        os.startfile(out_path)
except Exception:
    print("Could not auto-open — please open the file manually.")

plt.show()
