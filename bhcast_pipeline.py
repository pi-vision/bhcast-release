"""
BHCast end-to-end pipeline.

Steps:
1. Load an image (H, W) from disk.
2. Take the GRMHD dynamics model
   (via GRMHDEvaluator) to autoregressively forecast the next 60 frames.
3. Run the plasma feature extractors (`analyze_segment`) on the forecast.
4. Feed the resulting feature vector into a trained XGBoost classifier to
   predict the joint (spin, inclination) label.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tempfile
from typing import Dict, List, Tuple, Optional, Union

import imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import curve_fit, least_squares
from tqdm import tqdm
import numpy.fft as fft
import scipy.ndimage as ndimage
import traceback
from data_processor import DefaultDataProcessor

# Inline UNet definition for standalone use
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net model for image-to-image prediction."""
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class GRMHDInference:
    """
    Simplified inference class for GRMHD models. Performs autoregressive prediction
    from a single input image without requiring a dataloader or dataset.
    
    This class is designed for lightweight inference scenarios where you only need
    to generate predictions from a single starting frame. Only supports UNet models.
    """
    def __init__(self,
                 checkpoint_path: str,
                 data_processor_path: str,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 device: str = 'cuda'):
        """
        Initialize the inference class.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint file (.pt).
            data_processor_path (str): Path to the saved data processor file (.pkl).
            in_channels (int): Number of input channels (default 1).
            out_channels (int): Number of output channels (default 1).
            device (str): Device to run inference on ('cuda' or 'cpu').
        """
        self.device = device
        
        # Load model
        self.model = UNet(n_channels=in_channels, n_classes=out_channels)
        self.model = self.model.to(device)
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model checkpoint from {checkpoint_path}")
        

        # Load normalizers separately (expects files 'in_normalizer.pth' and 'out_normalizer.pth' in the same directory)
        import pickle

        data_processor_dir = os.path.dirname(data_processor_path)
        in_norm_path = os.path.join(data_processor_dir, "in_normalizer.pth")
        out_norm_path = os.path.join(data_processor_dir, "out_normalizer.pth")

        if not os.path.exists(in_norm_path):
            raise FileNotFoundError(f"in_normalizer not found at {in_norm_path}")
        if not os.path.exists(out_norm_path):
            raise FileNotFoundError(f"out_normalizer not found at {out_norm_path}")

        in_normalizer = torch.load(in_norm_path, map_location=device, weights_only=False)
        out_normalizer = torch.load(out_norm_path, map_location=device, weights_only=False)

        self.data_processor = DefaultDataProcessor(in_normalizer=in_normalizer, out_normalizer=out_normalizer)
        self.data_processor.to(device)
        print(f"Instantiated DataProcessor class with normalizers from {in_norm_path} and {out_norm_path}")

        
        self.model.eval()
    
    def predict(self, frame: Union[np.ndarray, torch.Tensor], horizon: int = 100) -> np.ndarray:
        """
        Generate autoregressive prediction from a single input frame.
        
        Args:
            frame (Union[np.ndarray, torch.Tensor]): Input frame of shape (H, W) or (C, H, W).
            horizon (int): Number of frames to predict (default 100).
            
        Returns:
            np.ndarray: Predicted frames of shape (horizon, H, W).
        """
        # Convert to torch tensor and ensure correct shape
        if not isinstance(frame, torch.Tensor):
            frame_tensor = torch.from_numpy(frame).float()
        else:
            frame_tensor = frame.float()
        
        # Ensure shape is (1, 1, H, W) or (1, C, H, W)
        if frame_tensor.ndim == 2:
            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.unsqueeze(0)  # (1, C, H, W)
        elif frame_tensor.ndim == 4:
            # If already batched, take first item
            frame_tensor = frame_tensor[0:1]
        else:
            raise ValueError(f"Unsupported frame shape: {frame_tensor.shape}. Expected (H, W), (C, H, W), or (1, C, H, W)")
        
        # Normalize the starting frame
        frame_tensor = frame_tensor.to(self.device)
        input_normalized = self.data_processor.in_normalizer.transform(frame_tensor)
        input_normalized = input_normalized.to(self.device)
        
        prediction_frames = []
        current_frame = input_normalized
        
        with torch.no_grad():
            for i in range(horizon):
                # Predict next frame (output is already normalized)
                prediction = self.model(current_frame)
                
                # Denormalize and store for output
                prediction_denorm = self.data_processor.out_normalizer.inverse_transform(prediction)
                prediction_frames.append(prediction_denorm.squeeze().cpu().numpy())
                
                # Use normalized prediction directly as input for next step (autoregressive)
                current_frame = prediction.detach()
        
        return np.array(prediction_frames)  # Shape: (horizon, H, W)


def calibrate_ring_location_and_radius(image: np.ndarray, fov_m: float, 
                                        grad_threshold: float = 0.0,
                                        smooth_sigma: float = 1.0) -> Tuple[float, float, float]:
    """
    Calibrate ring location (center) and radius using weighted least squares circle fitting.
    Based on the procedure in ring_alignment.py.
    
    Args:
        image: 2D array (H, W) representing the image
        fov_m: Field of view in gravitational radii M
        grad_threshold: Minimum gradient magnitude threshold (default: 0.0 to use all edges)
        smooth_sigma: Gaussian smoothing sigma for gradient computation (default: 1.0)
    
    Returns:
        Tuple of (center_x, center_y, radius_m) where:
        - center_x, center_y are in pixel coordinates
        - radius_m is in gravitational radii M
    """
    H, W = image.shape
    
    # Normalize image
    img_min = image.min()
    img_max = image.max()
    if img_max > img_min:
        img_norm = (image - img_min) / (img_max - img_min)
    else:
        img_norm = image
    
    # Compute gradient magnitude (edge strength)
    gy, gx = np.gradient(img_norm)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Optionally smooth
    if smooth_sigma > 0:
        grad_mag = gaussian_filter(grad_mag, sigma=smooth_sigma)
    
    # Extract coordinates weighted by edge strength
    y, x = np.nonzero(grad_mag > grad_threshold)
    if len(x) == 0:
        # Fallback to default center if no edges found
        center_x = (W - 1) / 2.0
        center_y = (H - 1) / 2.0
        # Default radius ~5.2M in pixels
        dx_m_per_pix = fov_m / float(W)
        radius_m = 5.2
        return center_x, center_y, radius_m
    
    weights = grad_mag[y, x]
    
    # Weighted least squares circle fit
    def weighted_residuals(params, x_coords, y_coords, w):
        xc, yc, r = params
        return w * (np.sqrt((x_coords - xc)**2 + (y_coords - yc)**2) - r)
    
    # Initial guess
    x0_guess = np.mean(x)
    y0_guess = np.mean(y)
    r0_guess = (x.max() - x.min()) / 2
    
    try:
        p0 = [x0_guess, y0_guess, r0_guess]
        res = least_squares(weighted_residuals, p0, args=(x, y, weights))
        xc, yc, r_pix = res.x
    except Exception:
        # Fallback if optimization fails
        xc = x0_guess
        yc = y0_guess
        r_pix = r0_guess
    
    # Convert radius from pixels to M
    dx_m_per_pix = fov_m / float(W)
    radius_m = r_pix * dx_m_per_pix
    
    return float(xc), float(yc), float(radius_m)


def save_as_hdf5_sequence(movie,
                          output_dir,
                          file_prefix,
                          spin,
                          inclination,
                          fov_uas,
                          fov_m,
                          dt,
                          start_time=0.0):
    """
    Save a movie segment as a sequence of HDF5 files with required metadata.
    Matches the layout expected by both analyses.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if movie.ndim == 4:
        T, _, H, W = movie.shape
    elif movie.ndim == 3:
        T, H, W = movie.shape
    else:
        raise ValueError(f"Unsupported movie shape: {movie.shape}. Expected (T, H, W) or (T, C, H, W).")

    # M per pixel according to given FoV in M and width in pixels
    dx_m_per_pix = fov_m / W

    for i in tqdm(range(T), desc=f"Exporting {file_prefix} HDF5 sequence"):
        filename = os.path.join(output_dir, f"{file_prefix}_{i:04d}.h5")
        with h5py.File(filename, 'w') as hf:
            header_grp = hf.create_group('header')
            camera_grp = header_grp.create_group('camera')
            fluid_header_grp = hf.create_group('fluid_header')

            camera_grp.create_dataset('dx', data=dx_m_per_pix)
            camera_grp.create_dataset('fovx_dsource', data=fov_uas)
            camera_grp.create_dataset('fovy_dsource', data=fov_uas)
            camera_grp.create_dataset('thetacam', data=inclination)

            fluid_header_grp.create_dataset('a', data=spin)

            header_grp.create_dataset('t', data=start_time + i * dt)
            header_grp.create_dataset('scale', data=1.0)

            pol_data = np.zeros((W, H, 4), dtype=np.float32)
            frame = np.squeeze(movie[i, ...])
            pol_data[:, :, 0] = frame.T
            hf.create_dataset('pol', data=pol_data)


# Helper functions for cylinder analysis
def myinterp(dat, x, y):
    """Deals with edge cases by clamping indices to valid range"""
    H, W = dat.shape
    
    # Safety check for arrays that are too small
    if H < 2 or W < 2:
        # Fallback to nearest neighbor if array is too small for bilinear interpolation
        i = max(0, min(int(round(x)), H-1))
        j = max(0, min(int(round(y)), W-1))
        return dat[i, j]
    
    i = int(x-0.5)
    j = int(y-0.5)
    
    # Clamp indices to ensure i+1 and j+1 are within bounds
    i = max(0, min(i, H-2))
    j = max(0, min(j, W-2))
    
    # Bilinear interpolation
    di = x-i
    dj = y-j
    
    # Clamp di and dj to [0, 1] to handle edge cases
    di = max(0.0, min(1.0, di))
    dj = max(0.0, min(1.0, dj))
    
    z00 = dat[i,j]
    z01 = dat[i,j+1]
    z10 = dat[i+1,j]
    z11 = dat[i+1,j+1]
    idat = z00*(1.-di)*(1.-dj) + z01*(1.-di)*dj + z10*di*(1.-dj) + z11*di*dj
    return idat


def mean_subtract(m, ntheta, nftot):
    """Subtract mean from each row and column"""
    for i in range(ntheta):
        m[:,i] = (m[:,i] - m[:,i].mean())
    for i in range(nftot):
        m[i,:] = (m[i,:] - m[i,:].mean())


def normalize(m):
    """Normalize cylinder data by subtracting mean from each row and column"""
    ntheta = m.shape[1]
    nftot = m.shape[0]
    for i in range(ntheta):
        m[:,i] = (m[:,i] - m[:,i].mean())
    for i in range(nftot):
        m[i,:] = (m[i,:] - m[i,:].mean())


def cross_correlate(c1, c2):
    """Calculate cross-correlation between two cylinder plots"""
    c1 -= np.mean(c1)
    c2 -= np.mean(c2)
    q1 = fft.fft2(c1)
    q2 = fft.fft2(c2)
    cross_corr = np.abs(fft.ifft2(q1*np.conj(q2)))
    return cross_corr[0]/(c1.shape[0]*c1.shape[1]*np.std(c1)*np.std(c2))


def parabola(x, a, b, c):
    """Parabola function for fitting"""
    return a*x**2 + b*x + c


def find_max(array, indices=None, window=5):
    """Find maximum location using parabolic fitting"""
    max_loc = np.argmax(array)
    midpoint = int(len(array)/2)
    try:
        if indices is None:
            indices = np.array(range(len(array)))
    except:
        pass
    
    a_roll = np.roll(array, midpoint - max_loc)
    ind_roll = np.roll(indices, midpoint - max_loc)

    a_w = a_roll[midpoint - window:midpoint + window]
    i_w = np.array(range(-window, window))
    p0 = (-1, 2*indices[midpoint], np.max(array) - indices[midpoint]**2)
    popt, pcov = curve_fit(parabola, i_w, a_w, p0=p0)

    fit_max_loc = max_loc - popt[1]/(2*popt[0])
    fl_low = int(np.floor(fit_max_loc))
    fl_high = int(np.ceil(fit_max_loc))
    if fl_high == len(indices):
        fit_max = indices[fl_low]
    elif fl_low == len(indices):
        fit_max = indices[fl_low - 1]
    else:
        fit_max = indices[fl_low] + (fit_max_loc - fl_low)*(indices[fl_high] - indices[fl_low])

    return fit_max


def run_cylinder_analysis(h5_directory, spin, inclination, fov_m, radius_scale_factors=None, start_index: int = None, num_frames: int = None, 
                          calibrated_center_x=None, calibrated_center_y=None, calibrated_radius_m=None):
    """
    Run the cylinder analysis on HDF5 files in the given directory.
    Returns the pattern speed and other relevant results.

    Args:
        h5_directory (str): Directory containing per-frame HDF5 files.
        spin (float): Black hole spin parameter.
        inclination (float): Inclination in degrees.
        fov_m (float): Field of view in units of M.
        radius_scale_factors (float | list[float] | None):
            Optional scalar or list of scalars to scale the baseline sampling radius r_pix.
            If None, defaults to [1.0].
        calibrated_center_x (float | None): Calibrated center x coordinate in pixels. If None, uses default calculation.
        calibrated_center_y (float | None): Calibrated center y coordinate in pixels. If None, uses default calculation.
        calibrated_radius_m (float | None): Calibrated baseline radius in M. If None, uses sqrt(27)*M as default.
    """
    try:
        # Load the files
        files = [f for f in os.listdir(h5_directory) if 'h5' in f]
        all_sorted = sorted(files)

        # Select a window if requested
        if start_index is not None and num_frames is not None:
            start = int(start_index)
            end = start + int(num_frames)
            sfiles = all_sorted[start:end]
        else:
            sfiles = all_sorted

        nftot = len(sfiles)
        if nftot < 4:
            print(f"Insufficient frames for dt estimate: {nftot} < 4")
            return None

        # Append directory path to each file name
        slash='/'
        full_file_names = []
        for i in np.arange(0, len(sfiles), 1):
           file_names = h5_directory+slash+sfiles[i]
           full_file_names.append(file_names)
        sfiles = full_file_names

        # Set some parameters for the run, based on first file in sequence
        hfp = h5py.File(sfiles[0],'r')
        FOV_M = fov_m
        FOV_uas = hfp['header']['camera']['fovx_dsource'][()]
        if hfp['header']['camera']['fovx_dsource'][()] != hfp['header']['camera']['fovy_dsource'][()]:
            print("Error: need a square image. This image appears to be {0:n} x {1:n} pixels".format(hfp['header']['camera']['fovx_dsource'][()], hfp['header']['camera']['fovy_dsource'][()]))
            return None
        scale = hfp['header']['scale'][()]
        imagep = np.copy(hfp['pol']).transpose((1,0,2))
        N = imagep.shape[0]
        M = imagep.shape[1]
        dx = FOV_uas/N
        dy = FOV_uas/M
        half_FOV = FOV_uas/2.

        # Find dt by reading in header/t from the first couple files
        file1 = sfiles[0]
        file2 = sfiles[1]
        file3 = sfiles[2]
        file4 = sfiles[3]

        hf1 = h5py.File(file1, 'r')
        hf2 = h5py.File(file2, 'r')
        hf3 = h5py.File(file3, 'r')
        hf4 = h5py.File(file4, 'r')

        dt2 = hf2[('header/t')][()] - hf1[('header/t')][()]
        dt3 = hf3[('header/t')][()] - hf2[('header/t')][()]
        dt4 = hf4[('header/t')][()] - hf3[('header/t')][()]
        dt = (dt2 + dt3 + dt4)/3
        hf1.close()
        hf2.close()
        hf3.close()
        hf4.close()

        dt = round(dt, 1)
        Tmax = nftot*dt

        # Allocate space for 2-D Stokes I image for each file
        Iall = np.zeros((N, M, nftot))

        # Loop over all the files to define Iall
        n = 0
        print("Reading files from " + str(h5_directory) + "...")
        for fname in tqdm(sfiles):
            hfp = h5py.File(fname,'r')
            imagep = np.copy(hfp['pol']).transpose((1,0,2))
            Iall[:,:,n] = imagep[:,:,0]
            hfp.close()
            n += 1

        # Creating Unnormalized Cylinder Plots
        print("Calculating Cylinder Plots...")

        # Smooth
        sig = 20/(dx *(2*np.sqrt(2*np.log(2))))
        sIall = ndimage.gaussian_filter(Iall, sigma=(sig,sig,0))

        # Make circle
        ntheta = 180
        theta = np.linspace(0.,2.*np.pi,ntheta,endpoint=False)
        dtheta = 360/ntheta
        pa = theta + 0.5*np.pi

        mean_im = np.mean(Iall, axis = 2)

        # Use calibrated center if provided, otherwise use default calculation
        if calibrated_center_x is not None and calibrated_center_y is not None:
            x0 = calibrated_center_x
            y0 = calibrated_center_y
        else:
            x0 = N/2
            y0 = M/2

        # Normalize radius_scale_factors to a list
        if radius_scale_factors is None:
            factors = [1.0]
        elif isinstance(radius_scale_factors, (int, float)):
            factors = [float(radius_scale_factors)]
        else:
            factors = [float(f) for f in radius_scale_factors]

        pattern_speed_by_factor = {}
        last_q = None
        last_qs = None
        last_qn = None
        last_qsn = None
        last_racf = None
        last_non_zero_pixels = None

        # Determine baseline radius in M
        if calibrated_radius_m is not None:
            baseline_radius_m = calibrated_radius_m
        else:
            baseline_radius_m = np.sqrt(27)

        print("Calculating Autocorrelations across radius scale factors ...")
        for fscale in factors:
            r_pix = fscale * baseline_radius_m * (N/FOV_M)
            icirc = y0 - r_pix*np.sin(pa)
            jcirc = x0 + r_pix*np.cos(pa)
            x = (jcirc - N/2.)*dx
            y = (icirc - M/2.)*dy

            # Create Unsmoothed Unnormalized Cylinder Plot
            q = np.zeros((nftot,ntheta))
            for k in range(nftot):
                for l in range(ntheta):
                    q[k,l] = myinterp(Iall[:,:,k], icirc[l], jcirc[l])

            # Create Smoothed Unnormalized Cylinder Plot
            qs = np.zeros((nftot,ntheta))
            for k in range(nftot):
                for l in range(ntheta):
                    qs[k,l] = myinterp(sIall[:,:,k], icirc[l], jcirc[l])

            # Creating Normalized Cylinder Plots
            qn =  np.copy(q)
            qn = np.log10(qn)
            mean_subtract(qn, ntheta, nftot)

            qsn =  np.copy(qs)
            qsn = np.log10(qsn)
            mean_subtract(qsn, ntheta, nftot)

            # Producing Autocorrelation Function
            qk = fft.fft2(qsn)
            Pk = np.absolute(qk)**2
            acf = np.real( fft.ifft2(Pk) )
            acf = acf/acf[0,0]

            # Shift so that the peak correlation is in the middle
            shifti = int(acf.shape[0]/2.)
            shiftj = int(acf.shape[1]/2.)
            racf = np.roll(acf, (shifti, shiftj), axis=(0,1))

            # Normalize the Autocorrelation
            racf /= np.max(racf)
            racf_cut = np.copy(racf)

            # Calculate the 2nd Moments
            ts = np.linspace(-len(racf)/2, len(racf)/2, len(racf), endpoint = False)
            phis = np.linspace(-len(racf[0])/2, len(racf[0])/2, len(racf[0]), endpoint = False)
            delta_t = ts[1] - ts[0]
            delta_phi = phis[1] - phis[0]

            moment_t = 0
            moment_phi = 0
            moment_t_phi = 0
            moment = 0

            xi_crit = 0.8
            for i in range(len(ts)):
                for j in range(len(phis)):
                    if racf[i,j] < xi_crit:
                        racf_cut[i,j] = 0.0
                        continue

                    moment += racf_cut[i,j]
                    moment_t += racf_cut[i,j] * ts[i] * ts[i]
                    moment_phi += racf_cut[i,j] * phis[j] * phis[j]
                    moment_t_phi += racf_cut[i,j] * ts[i] * phis[j]

            non_zero_pixels = np.count_nonzero(racf_cut)
            if non_zero_pixels < 20:
                print("Caution: calculating moments over fewer than 20 pixels! Consider setting a lower xi_crit threshold")

            # Correct 2nd moment for units and normalize with 1st moment
            moment *= delta_t*delta_phi
            moment_t *= delta_t*delta_phi/moment
            moment_phi *= delta_t*delta_phi/moment
            moment_t_phi *= delta_t*delta_phi/moment

            # Find the pattern speed from the moments
            pattern_speed = moment_t_phi/moment_t
            pattern_speed = pattern_speed*dtheta/dt

            pattern_speed_by_factor[fscale] = pattern_speed

            # keep last intermediates for optional return/debugging
            last_q = q
            last_qs = qs
            last_qn = qn
            last_qsn = qsn
            last_racf = racf
            last_non_zero_pixels = non_zero_pixels

        # Determine a canonical pattern_speed to expose (prefer factor=1.0)
        canonical_ps = None
        if 1.0 in pattern_speed_by_factor:
            canonical_ps = pattern_speed_by_factor[1.0]
        elif len(pattern_speed_by_factor) == 1:
            canonical_ps = next(iter(pattern_speed_by_factor.values()))

        print("################### Result ###########################")
        print("Parameters:\nspin: + {0:0.1f}\ninclination: {1:n}".format(spin, inclination))
        print("######################################################")
        if canonical_ps is not None:
            print("Pattern Speed (factor 1.0):\n{0:0.2f} deg/GMc^-3".format(canonical_ps))
        print("Factors computed:", sorted(pattern_speed_by_factor.keys()))

        return {
            'pattern_speed': canonical_ps,
            'pattern_speed_by_factor': pattern_speed_by_factor,
            'spin': spin,
            'inclination': inclination,
            'nftot': nftot,
            'dt': dt,
            'ntheta': ntheta,
            'dtheta': dtheta,
            'xi_crit': xi_crit,
            'non_zero_pixels': last_non_zero_pixels,
            'raw_cylinder': last_q,
            'smoothed_cylinder': last_qs,
            'normalized_cylinder': last_qn,
            'smoothed_normalized_cylinder': last_qsn,
            'autocorrelation': last_racf
        }

    except Exception as e:
        print(f"Error in cylinder analysis: {e}")
        print(traceback.format_exc())
        return None


def calc_spiral_angles_for_movie(h5_directory, spin, inclination, fov_m=None, fov_uas=None, nftot=None, start_index: int = None,
                                  calibrated_center_x=None, calibrated_center_y=None, calibrated_radius_m=None):
    """
    Calculate spiral arm pitch angles for a movie sequence using HDF5 files.
    
    Args:
        h5_directory: Directory containing HDF5 files
        spin: black hole spin parameter
        inclination: viewing inclination in degrees
        fov_m: field of view in gravitational radii M (if None, will read from H5)
        fov_uas: field of view in microarcseconds (if None, will read from H5)
        nftot: number of frames to use (if None, will use all frames)
        calibrated_center_x (float | None): Calibrated center x coordinate in pixels. If None, uses default calculation.
        calibrated_center_y (float | None): Calibrated center y coordinate in pixels. If None, uses default calculation.
        calibrated_radius_m (float | None): Calibrated baseline radius in M. If None, uses sqrt(27)*M as default.
    
    Returns:
        Dictionary with pitch angle and correlation
    """
    try:
        # Loading files
        files = [f for f in os.listdir(h5_directory) if 'h5' in f]
        all_sorted = sorted(files)
        nftot_available = len(all_sorted)

        # Determine window
        if nftot is None:
            nftot = nftot_available

        nftot = min(nftot, nftot_available)

        if start_index is None:
            sfiles = all_sorted[-nftot:]
            start_used = max(0, nftot_available - nftot)
        else:
            start_used = int(start_index)
            end_used = start_used + int(nftot)
            sfiles = all_sorted[start_used:end_used]

        if len(sfiles) < 4:
            print(f"Insufficient frames for dt estimate: {len(sfiles)} < 4")
            return None

        print('first file: ', sfiles[0], 'last file: ', sfiles[-1])
        # Append directory path to each file name
        slash='/'
        full_file_names = []
        for k in np.arange(0, len(sfiles), 1):
           file_names = h5_directory+slash+sfiles[k]
           full_file_names.append(file_names)
        sfiles = full_file_names

        # Set some parameters for the run, based on first file in sequence
        hfp = h5py.File(sfiles[0],'r')
        FOV_M = hfp['header']['camera']['dx'][()]
        FOV_uas = hfp['header']['camera']['fovx_dsource'][()]
        if hfp['header']['camera']['fovx_dsource'][()] != hfp['header']['camera']['fovy_dsource'][()]:
            print("Error: need a square image. This image appears to be {0:n} x {1:n} pixels".format(hfp['header']['camera']['fovx_dsource'][()], hfp['header']['camera']['fovy_dsource'][()]))
            return None
        scale = hfp['header']['scale'][()]
        imagep = np.copy(hfp['pol'])
        N = imagep.shape[0]
        M = imagep.shape[1]
        dx = FOV_uas/N
        dy = FOV_uas/M
        half_FOV = FOV_uas/2.

        # Set dt from header
        file1 = sfiles[0]
        file2 = sfiles[1]
        file3 = sfiles[2]
        file4 = sfiles[3]

        hf1 = h5py.File(file1, 'r')
        hf2 = h5py.File(file2, 'r')
        hf3 = h5py.File(file3, 'r')
        hf4 = h5py.File(file4, 'r')

        dt2 = hf2[('header/t')][()] - hf1[('header/t')][()]
        dt3 = hf3[('header/t')][()] - hf2[('header/t')][()]
        dt4 = hf4[('header/t')][()] - hf3[('header/t')][()]
        dt = (dt2 + dt3 + dt4)/3
        hf1.close()
        hf2.close()
        hf3.close()
        hf4.close()

        dt = round(dt, 1)
        Tmax = nftot*dt

        # Allocate space
        Iall = np.zeros((N, M, nftot))

        # Read in files
        n = 0
        for fname in sfiles:
            hfp = h5py.File(fname,'r')
            imagep = np.copy(hfp['pol'])
            Iall[:,:,n] = imagep[:,:,0]
            hfp.close()
            n += 1

        # Process unblurred case only
        blurred = False
        if blurred:
            sig = 20/(dx *(2*np.sqrt(2*np.log(2))))
        else:
            sig = 2/(dx *(2*np.sqrt(2*np.log(2))))

        sIall = ndimage.gaussian_filter(Iall, sigma=(sig,sig,0))
        cylinders = {}
        cylinders_standardized = {}
        
        # Creating Smoothed Unnormalized Cylinder Plots
        ntheta = 720
        theta = np.linspace(0.,2.*np.pi,ntheta)
        dtheta = 360/ntheta
        pa = theta + 0.5*np.pi

        # Use calibrated center if provided, otherwise use default calculation
        if calibrated_center_x is not None and calibrated_center_y is not None:
            x0 = calibrated_center_x
            y0_base = calibrated_center_y
        else:
            M_x0 = 5
            x0 = N/2
            y0_base = M/2

        # Determine baseline radius in M for scaling
        if calibrated_radius_m is not None:
            baseline_radius_m = calibrated_radius_m
        else:
            baseline_radius_m = np.sqrt(27)
        
        # Use radius=1.0 and delta_r=0.1
        r_inner = 1.0
        delta_r = 0.1
        radii = [r_inner, r_inner + delta_r]
        
        M_x0 = 5
        
        for radius in radii:
            y0 = y0_base
            r_pix = radius * baseline_radius_m * M_x0 / dx
            icirc = y0 + r_pix*np.sin(pa)
            jcirc = x0 + r_pix*np.cos(pa)
            x = (jcirc - N/2.)*dx
            y = (icirc - M/2.)*dy

            # Create Smoothed Unnormalized Cylinder Plot
            q = np.zeros((nftot,ntheta))
            for k in range(nftot):
                for l in range(ntheta):
                    q[k,l] = myinterp(sIall[:,:,k], icirc[l], jcirc[l])
            q[q < 1E-9] = 1E-9
            cylinders[radius] = np.copy(q)
            qn =  np.copy(np.log(q))
            normalize(qn)
            cylinders_standardized[radius] = qn

        # Calculate cross-correlation and pitch angle
        r1 = radii[0]
        r2 = radii[1]
        c1 = cylinders_standardized[r1]
        c2 = cylinders_standardized[r2]
        xi = cross_correlate(c1, c2)

        thetas = np.linspace(0, 360, ntheta)
        shift = find_max(xi, indices = thetas)
        corr = np.max(xi)
        dphi = np.pi/180.*shift
        if dphi > np.pi:
            dphi = dphi - 2*np.pi
        dr = np.diff(np.log(radii))[0]

        pitch_angle = 180./np.pi*np.arctan2(dphi, dr)

        return {
            'pitch_angle': pitch_angle,
            'correlation': corr,
            'nftot_used': nftot,
            'start_frame': int(start_used),
            'FOV_M': FOV_M,
            'FOV_uas': FOV_uas,
            'N': N,
            'M': M
        }

    except Exception as e:
        print(f"Error calculating spiral angles: {e}")
        print(traceback.format_exc())
        return None


def analyze_segment(h5_dir,
                    spin,
                    inclination,
                    fov_m,
                    nftot_spiral=None,
                    segment_frames_np=None,
                    segment_interval=1,
                    r_interest_m=5.2,
                    n_angles=360,
                    radius_factors=None,
                    start_index: int = None,
                    segment_length: int = None,
                    calibrate_ring=False,
                    grad_threshold=0.0,
                    smooth_sigma=1.0,
                    features_to_calculate: Optional[set] = None):
    """
    Run pattern speed, spiral pitch angle, and asymmetry analyses against a prepared HDF5 directory.
    
    Args:
        features_to_calculate: Optional set of feature names to calculate. Can include:
            - 'pattern_speed': Calculate pattern speed and pattern speed slope
            - 'spiral': Calculate spiral pitch angle, correlation, and nftot_used
            - 'asymmetry': Calculate asymmetry metrics
        If None (default), all features are calculated.
    
    Returns:
        dict with calculated measurements. Features not in features_to_calculate are set to None.
    """
    # Default: calculate all features if not specified
    if features_to_calculate is None:
        features_to_calculate = {'pattern_speed', 'spiral', 'asymmetry'}
    else:
        # Convert to set if list or tuple provided
        features_to_calculate = set(features_to_calculate)
    
    # Calibrate ring location and radius if requested (needed for pattern_speed or spiral)
    calibrated_center_x = None
    calibrated_center_y = None
    calibrated_radius_m = None
    
    if calibrate_ring and segment_frames_np is not None and segment_frames_np.ndim == 3:
        try:
            mean_image = np.mean(segment_frames_np, axis=0)
            center_x, center_y, radius_m = calibrate_ring_location_and_radius(
                mean_image, fov_m, grad_threshold=grad_threshold, smooth_sigma=smooth_sigma
            )
            calibrated_center_x = center_x
            calibrated_center_y = center_y
            calibrated_radius_m = radius_m
        except Exception as e:
            # If calibration fails, use defaults
            pass
    
    # Initialize output dict
    out = {}
    
    # Pattern speed analysis
    if 'pattern_speed' in features_to_calculate:
        if radius_factors is not None:
            if 1.0 not in radius_factors:
                radius_factors.append(1.0)
        ps_result = run_cylinder_analysis(
            h5_dir,
            spin,
            inclination,
            fov_m,
            radius_scale_factors=radius_factors,
            start_index=start_index,
            num_frames=segment_length,
            calibrated_center_x=calibrated_center_x,
            calibrated_center_y=calibrated_center_y,
            calibrated_radius_m=calibrated_radius_m,
        )
        
        if ps_result is not None:
            out['pattern_speed'] = ps_result.get('pattern_speed')
            out['pattern_speed_by_factor'] = ps_result.get('pattern_speed_by_factor')
        else:
            out['pattern_speed'] = None
            out['pattern_speed_by_factor'] = None
        
        # Get cylinder plot if available
        if ps_result is not None:
            out['cylinder'] = ps_result.get('smoothed_normalized_cylinder')
        
        # If multiple radius factors are provided, compute slope d(pattern_speed)/d(factor)
        try:
            ps_by_factor = out.get('pattern_speed_by_factor')
            slope = None
            if isinstance(ps_by_factor, dict) and len(ps_by_factor) >= 2:
                xs = sorted(ps_by_factor.keys())
                ys = [ps_by_factor[x] for x in xs]
                # Ensure finite values
                xs_arr = np.array(xs, dtype=float)
                ys_arr = np.array(ys, dtype=float)
                mask = np.isfinite(xs_arr) & np.isfinite(ys_arr)
                if np.count_nonzero(mask) >= 2:
                    coeffs = np.polyfit(xs_arr[mask], ys_arr[mask], 1)
                    slope = float(coeffs[0])
            out['pattern_speed_slope'] = slope
        except Exception:
            out['pattern_speed_slope'] = None
    else:
        out['pattern_speed'] = None
        out['pattern_speed_by_factor'] = None
        out['pattern_speed_slope'] = None
    
    # Spiral pitch angle analysis
    if 'spiral' in features_to_calculate:
        spiral_result = calc_spiral_angles_for_movie(
            h5_dir,
            spin,
            inclination,
            nftot=(segment_length if nftot_spiral is None else min(nftot_spiral, segment_length if segment_length is not None else nftot_spiral)),
            start_index=start_index,
            calibrated_center_x=calibrated_center_x,
            calibrated_center_y=calibrated_center_y,
            calibrated_radius_m=calibrated_radius_m,
        )
        
        if spiral_result is not None:
            out['pitch_angle_deg'] = spiral_result.get('pitch_angle')
            out['corr'] = spiral_result.get('correlation')
            out['nftot_used'] = spiral_result.get('nftot_used')
        else:
            out['pitch_angle_deg'] = None
            out['corr'] = None
            out['nftot_used'] = None
    else:
        out['pitch_angle_deg'] = None
        out['corr'] = None
        out['nftot_used'] = None

    # Asymmetry computation on the provided segment (time-averaged angular profile at fixed radius)
    if 'asymmetry' in features_to_calculate:
        try:
            if isinstance(segment_frames_np, torch.Tensor):
                segment_frames_np = segment_frames_np.cpu().numpy()
            if segment_frames_np.ndim == 4:
                segment_frames_np = segment_frames_np.squeeze()
            Tseg, H, W = segment_frames_np.shape
            # Convert desired radius in M to pixels
            # dx (M per pixel) = fov_m / W
            dx_m_per_pix = fov_m / float(W)
            
            # Use calibrated values if provided, otherwise use defaults
            if calibrated_center_x is not None and calibrated_center_y is not None:
                center_x = calibrated_center_x
                center_y = calibrated_center_y
            else:
                # Center at the image midpoint
                center_x = (W - 1) / 2.0
                center_y = (H - 1) / 2.0
            
            # Use calibrated radius if provided, otherwise use default r_interest_m
            if calibrated_radius_m is not None:
                r_pix = calibrated_radius_m / dx_m_per_pix
            else:
                r_pix = r_interest_m / dx_m_per_pix

            thetas = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
            x_coords = center_x + r_pix * np.cos(thetas)
            y_coords = center_y + r_pix * np.sin(thetas)
            sample_coords = np.vstack([y_coords, x_coords])

            all_profiles = []
            for frame in segment_frames_np:
                intensity_profile = map_coordinates(frame, sample_coords, order=3, mode='nearest')
                all_profiles.append(intensity_profile)

            mean_profile = np.mean(np.asarray(all_profiles), axis=0)

            def cos_model(theta, A, theta_0, C):
                return A * np.cos(theta + theta_0) + C

            guess_A = (np.max(mean_profile) - np.min(mean_profile)) / 2.0
            guess_C = float(np.mean(mean_profile))
            guess_theta_0 = float(thetas[int(np.argmax(mean_profile))])
            p0 = [guess_A, guess_theta_0, guess_C]

            popt, pcov = curve_fit(cos_model, thetas, mean_profile, p0=p0, maxfev=10000)
            A_fit, theta0_fit, C_fit = [float(v) for v in popt]

            asymmetry = float(A_fit / C_fit) if C_fit != 0.0 else None
            asymmetry = abs(asymmetry)

            out['asymmetry_A_over_C'] = asymmetry
            out['asymmetry_A_fit'] = A_fit
            out['asymmetry_C_fit'] = C_fit
            out['asymmetry_theta0_fit'] = theta0_fit
            print('Generated asymmetry successfully with value: ', asymmetry)

        except Exception:
            print('Error computing asymmetry')
            out['asymmetry_A_over_C'] = None
            out['asymmetry_A_fit'] = None
            out['asymmetry_C_fit'] = None
            out['asymmetry_theta0_fit'] = None
    else:
        out['asymmetry_A_over_C'] = None
        out['asymmetry_A_fit'] = None
        out['asymmetry_C_fit'] = None
        out['asymmetry_theta0_fit'] = None

    return out


ALLOWED_INCLINATIONS = {
    'sgra': [10, 30, 50, 70, 110, 130, 150, 170],
}

DEFAULT_FEATURE_ORDER = [
    'pattern_speed',
    'pitch_angle_deg',
    'asymmetry_A_over_C',
    'pattern_speed_slope',
]

def default_normalization_regex(dataset_name: str) -> str:
    if dataset_name == 'sgra':
        return r'(.*)(?:Ma)(.*)(?:_Rh10_i(10|30|50|70|110|130|150|170)_w4\.pt)'
    if dataset_name == 'm87':
        return r'(.*)(?:Ma)(.*)(?:_Rh10_Rl10_i(17|163)_w4\.pt)'
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def load_image_file(path: str) -> np.ndarray:
    """
    Load a single image file as a (H, W) float32 numpy array.
    Supported formats: .png, .jpg, .jpeg, .npy
    
    For .npy files, if the array has multiple frames, uses the first frame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        # Load numpy array
        data = np.load(path)
        # If multi-dimensional, take first frame
        if data.ndim == 2:
            # Already 2D (H, W)
            frame = data
        elif data.ndim == 3:
            # Shape (T, H, W) - take first frame
            frame = data[0]
        elif data.ndim == 4:
            # Shape (T, C, H, W) - take first frame, then first channel or squeeze
            frame = data[0]  # Now shape (C, H, W)
            if frame.shape[0] == 1:
                # Single channel, squeeze it
                frame = frame.squeeze(0)  # Now (H, W)
            else:
                # Multiple channels, take first channel
                frame = frame[0]  # Now (H, W)
        else:
            raise ValueError(f"Unsupported .npy array shape: {data.shape}. Expected 2D, 3D (T, H, W), or 4D (T, C, H, W)")
        return frame.astype(np.float32)
    elif ext in {'.png', '.jpg', '.jpeg'}:
        with Image.open(path) as img:
            frame = np.array(img.convert('F'), dtype=np.float32)
        return frame.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image format: {ext}. Supported: .png, .jpg, .jpeg, .npy")


def save_forecast_comparison_gif(
    truth: np.ndarray,
    forecast: np.ndarray,
    output_path: str,
    fps: int = 10,
) -> None:
    num_frames = min(len(truth), len(forecast))
    frames: List[np.ndarray] = []

    for idx in range(num_frames):
        truth_frame = truth[idx]
        forecast_frame = forecast[idx]

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        for ax, data, title in zip(
            axes,
            [truth_frame, forecast_frame],
            [f"Truth #{idx}", f"Forecast #{idx}"],
        ):
            im = ax.imshow(data, cmap='inferno')
            ax.axis('off')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        frames.append(image)

    if not frames:
        raise ValueError("No overlapping frames between truth and forecast for visualization.")

    imageio.mimsave(output_path, frames, fps=fps)


def write_forecast_to_hdf5(
    forecast: np.ndarray,
    output_dir: str,
    spin_meta: float,
    inclination_meta: int,
    fov_uas: float,
    fov_m: float,
    dt: float,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    save_as_hdf5_sequence(
        movie=forecast,
        output_dir=output_dir,
        file_prefix="frame",
        spin=spin_meta,
        inclination=inclination_meta,
        fov_uas=fov_uas,
        fov_m=fov_m,
        dt=dt,
        start_time=0.0,
    )
    return output_dir


def extract_features(
    h5_dir: str,
    forecast: np.ndarray,
    spin_meta: float,
    inclination_meta: int,
    fov_m: float,
    nftot_spiral: int | None,
    r_interest_m: float,
    n_angles: int,
    radius_factors: List[float] | None,
    calibrate_ring: bool,
    grad_threshold: float,
    smooth_sigma: float,
) -> Dict[str, float]:
    segment_length = forecast.shape[0]
    joint = analyze_segment(
        h5_dir=h5_dir,
        spin=spin_meta,
        inclination=inclination_meta,
        fov_m=fov_m,
        nftot_spiral=nftot_spiral,
        segment_frames_np=forecast,
        segment_interval=1,
        r_interest_m=r_interest_m,
        n_angles=n_angles,
        radius_factors=radius_factors,
        start_index=0,
        segment_length=segment_length,
        calibrate_ring=calibrate_ring,
        grad_threshold=grad_threshold,
        smooth_sigma=smooth_sigma,
    )
    return joint


def feature_vector_from_metrics(
    metrics: Dict[str, float],
    required_order: List[str],
) -> np.ndarray:
    vec = []
    for key in required_order:
        value = metrics.get(key)
        vec.append(float(value) if value is not None else np.nan)
    arr = np.asarray(vec, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_label_metadata(label_map_path: str) -> Tuple[Dict[int, Tuple[int, int]], List[str], Dict[int, float]]:
    import pickle

    with open(label_map_path, 'rb') as f:
        meta = pickle.load(f)

    index_to_pair = meta['index_to_pair']
    feature_names = meta.get('feature_names', DEFAULT_FEATURE_ORDER)
    spin_to_class = meta.get('spin_to_class')
    spin_values = meta.get('spin_values')

    if spin_to_class is None and spin_values is not None:
        spin_to_class = {float(s): i for i, s in enumerate(sorted(spin_values))}

    if spin_to_class is None:
        raise ValueError("Label map must include spin_to_class or spin_values metadata.")

    class_to_spin = {cls: float(spin) for spin, cls in spin_to_class.items()}
    return index_to_pair, feature_names, class_to_spin


def decode_prediction(
    pred_idx: int,
    index_to_pair: Dict[int, Tuple[int, int]],
    class_to_spin: Dict[int, float],
    dataset_name: str,
) -> Tuple[float | None, int | None]:
    if pred_idx not in index_to_pair:
        return None, None

    spin_cls, inc_cls = index_to_pair[pred_idx]
    spin_val = class_to_spin.get(spin_cls)

    inc_values = ALLOWED_INCLINATIONS.get(dataset_name, ALLOWED_INCLINATIONS['sgra'])
    inc_val = inc_values[inc_cls] if 0 <= inc_cls < len(inc_values) else None
    return spin_val, inc_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BHCast pipeline: image -> forecast -> plasma features -> XGBoost inference")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the UNet model checkpoint (.pt)")
    parser.add_argument("--data_processor_path", type=str, required=True, help="Path to the saved data processor (.pkl)")
    parser.add_argument("--dataset_name", type=str, choices=['sgra', 'm87'], default='sgra')
    parser.add_argument("--device", type=str, default='cuda', help="Device to run inference on ('cuda' or 'cpu')")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=1, help="Number of output channels")

    parser.add_argument("--forecast_horizon", type=int, default=60, help="Number of frames to autoregressively predict")
    parser.add_argument("--spin_hint", type=float, default=0.0, help="Spin metadata stored in temporary HDF5 files (not used for inference)")
    parser.add_argument("--inclination_hint", type=int, default=0, help="Inclination metadata stored in temporary HDF5 files (not used for inference)")
    parser.add_argument("--fov_uas", type=float, default=200.0, help="FoV in microarcseconds for HDF5 export")
    parser.add_argument("--fov_m_sgra", type=float, default=39.764)
    parser.add_argument("--fov_m_m87", type=float, default=52.355)
    parser.add_argument("--dt", type=float, default=5.0, help="Frame spacing used in HDF5 headers")

    parser.add_argument("--nftot_spiral", type=int, default=None, help="Optional cap on frames for spiral analysis")
    parser.add_argument("--r_interest_m", type=float, default=5.2, help="Radius for asymmetry sampling (in M)")
    parser.add_argument("--n_angles", type=int, default=360, help="Angular samples for asymmetry computation")
    parser.add_argument("--ps_radius_min", type=float, default=0.75)
    parser.add_argument("--ps_radius_max", type=float, default=1.5)
    parser.add_argument("--ps_radius_num", type=int, default=5)
    parser.add_argument("--calibrate_ring", action='store_true')
    parser.add_argument("--grad_threshold", type=float, default=0.0)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)

    parser.add_argument("--temp_dir", type=str, default=None, help="Optional directory to store intermediate HDF5 files")
    parser.add_argument("--keep_temp", action='store_true', help="Keep temporary directory for inspection")
    parser.add_argument("--comparison_path", type=str, default="bhcast_forecast_vs_truth.gif", help="Path to save truth vs forecast visualization")
    parser.add_argument("--comparison_fps", type=int, default=10, help="FPS for the comparison GIF")
    parser.add_argument("--cylinder_plot_path", type=str, default=None, help="Path to save the generated cylinder plot visualization")

    parser.add_argument("--xgb_model", type=str, required=True, help="Path to trained XGBoost .joblib file")
    parser.add_argument("--label_map", type=str, required=True, help="Path to label-map pickle saved during XGBoost training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load input image
    print(f"Loading image from {args.input_image}...")
    image = load_image_file(args.input_image)
    
    # Ensure image is 2D (H, W) - squeeze channel dimension if present
    if image.ndim == 3:
        image = image.squeeze(0)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image (H, W), got shape {image.shape}")

    # Initialize inference class
    print("Initializing GRMHD inference model...")
    inference = GRMHDInference(
        checkpoint_path=args.checkpoint_path,
        data_processor_path=args.data_processor_path,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        device=args.device,
    )

    # Generate forecast
    print(f"Generating {args.forecast_horizon} frame forecast...")
    forecast = inference.predict(frame=image, horizon=args.forecast_horizon)
    
    # Ensure forecast is (T, H, W)
    if forecast.ndim == 4:
        # Forecast is (T, C, H, W)
        if forecast.shape[1] > 1:
            forecast = forecast[:, 0]
        else:
            forecast = np.squeeze(forecast, axis=1)
    if forecast.ndim != 3:
        raise ValueError(f"Forecast must have shape (T, H, W); got {forecast.shape}")

    temp_root = args.temp_dir or tempfile.mkdtemp(prefix="bhcast_pipeline_")
    h5_dir = os.path.join(temp_root, "forecast_h5")
    fov_m = args.fov_m_sgra if args.dataset_name == 'sgra' else args.fov_m_m87

    write_forecast_to_hdf5(
        forecast=forecast,
        output_dir=h5_dir,
        spin_meta=args.spin_hint,
        inclination_meta=args.inclination_hint,
        fov_uas=args.fov_uas,
        fov_m=fov_m,
        dt=args.dt,
    )

    radius_factors = list(np.linspace(float(args.ps_radius_min), float(args.ps_radius_max), int(args.ps_radius_num)))
    metrics = extract_features(
        h5_dir=h5_dir,
        forecast=forecast,
        spin_meta=args.spin_hint,
        inclination_meta=args.inclination_hint,
        fov_m=fov_m,
        nftot_spiral=args.nftot_spiral,
        r_interest_m=args.r_interest_m,
        n_angles=args.n_angles,
        radius_factors=radius_factors,
        calibrate_ring=args.calibrate_ring,
        grad_threshold=args.grad_threshold,
        smooth_sigma=args.smooth_sigma,
    )

    index_to_pair, required_feature_order, class_to_spin = load_label_metadata(args.label_map)
    feature_vec = feature_vector_from_metrics(metrics, required_feature_order)

    model = joblib.load(args.xgb_model)
    probs = model.predict_proba(feature_vec.reshape(1, -1))[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    pred_spin, pred_inc = decode_prediction(pred_idx, index_to_pair, class_to_spin, args.dataset_name)

    spin_class = "prograde (a* > 0)" if pred_spin > 0 else "retrograde (a* < 0)"
    if pred_inc in [10, 30]:
        inc_class = "positive face-on (10/30)"
    elif pred_inc in [50, 70]:
        inc_class = "positive edge-on (50/70)"
    elif pred_inc in [110, 130]:
        inc_class = "negative edge-on (110/130)"
    elif pred_inc in [150, 170]:
        inc_class = "negative face-on (150/170)"
    else:
        inc_class = f"unknown ({pred_inc})"

    print("\n=== BHCast Pipeline Results ===")
    print(f"Input image: {args.input_image}")
    print(f"Forecast horizon: {args.forecast_horizon} frames")
    print("Plasma features:")
    for key, value in metrics.items():
        if key in DEFAULT_FEATURE_ORDER:
            print(f"  {key}: {value}")
    print("\nXGBoost prediction:")
    print(f"  Joint class index: {pred_idx}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Spin: {spin_class}")
    print(f"  Inclination: {inc_class}")

    if args.cylinder_plot_path and 'smoothed_normalized_cylinder' in metrics and metrics['smoothed_normalized_cylinder'] is not None:
        print(f"\nSaving cylinder plot to {args.cylinder_plot_path}...")
        cylinder = metrics['smoothed_normalized_cylinder']
        plt.figure(figsize=(10, 6))
        plt.imshow(cylinder, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.title('Smoothed Normalized Cylinder Plot')
        plt.xlabel('Angle Index')
        plt.ylabel('Time / Frame Index')
        plt.savefig(args.cylinder_plot_path)
        plt.close()

    if not args.keep_temp and args.temp_dir is None:
        shutil.rmtree(temp_root, ignore_errors=True)
    else:
        print(f"Temporary artifacts retained at: {temp_root}")



if __name__ == "__main__":
    main()

