from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.io import imread
from datetime import datetime
import numpy as np
import argparse
import os

init_cnt = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Network settings for OMEGA optically-collaborative ONN")
    # Supported tasks
    parser.add_argument("-p", "--platform", type = str, default = "Test", choices = ["BONN", "Test"])
    # Light path simulation settings, default values set by BONN tasks
    # Length argues are in mm
    parser.add_argument("-w", "--input_lambda", type = int, default = 658)          # Input wavelength
    parser.add_argument("-n", "--refractive_index", type = float, default = 1)      # Prop enviornment refractive index
    parser.add_argument("-f", "--filling_rate", type = float, default = 0.94)
    parser.add_argument("-r", "--input_Gauss_radius", type = float, default = 2.83)    # Input Gauss light radius, mm
    parser.add_argument('-sw', "--simulation_width", type = float, default = 20)     # Size of output observation window 
    parser.add_argument('-d_in', "--input_distance", type = int, default = 100)     # Prop distance from input to modulator
    parser.add_argument('-d_out', "--output_distance", type = int, default = 100)  # Prop distance from modulator to observer
    parser.add_argument('-d_layer', "--layer_distance", type = int, default = 80)   # Prop distance between modulators
    parser.add_argument("-layer_num", "--layer_number", type = int, default = 4)
    parser.add_argument("-dim_p", "--pixel_pitch", type = float, default = 13.8)    # Pixel size of modulator
    parser.add_argument("--expand_factor", type = int, default = 1)
    parser.add_argument("--upsample_factor", type = int, default = 1)
    parser.add_argument("--see_patterns", type = int, default = 0)
    
    args = parser.parse_args()
    return args

class modulators_info:
    def __init__(self, pattern, d_layer, dev_flag):
        self.pattern = pattern
        self.d_layer = d_layer
        self.dev_flag = dev_flag  # 1 for phase modulation, 0 for amplitude modulation

def gauss_beam_am(N_yy, N_xx, SLM_pixel_pitch_m, Input_Gauss_radius_m):
    """
    Generate Gaussian beam amplitude distribution
    """
    Input_beam_am = np.zeros((N_yy, N_xx))
    Center_yy = (N_yy + 1) / 2
    Center_xx = (N_xx + 1) / 2
    
    for ii in range(N_xx):
        Position_xx = abs(ii + 1 - Center_xx) * SLM_pixel_pitch_m
        for jj in range(N_yy):
            Position_yy = abs(jj + 1 - Center_yy) * SLM_pixel_pitch_m
            Input_beam_am[jj, ii] = np.exp(-((Position_xx**2 + Position_yy**2) / (2 * Input_Gauss_radius_m**2)))
    
    return Input_beam_am

def normxin(args):
    """
    Normalize input array to [0,1] range
    Handle case where all values are the same
    """
    min_val = np.min(args)
    max_val = np.max(args)
    
    # Handle case where all values are the same
    if max_val - min_val == 0:
        return np.zeros_like(args)
    else:
        output_args = (args - min_val) / (max_val - min_val)
        return output_args

def norm2D(args):
    """
    Normalize 2D or 3D array
    If 2D, normalize directly
    If 3D, normalize each slice separately
    """
    if len(args.shape) == 2:
        # If 2D array, apply normalization directly
        return normxin(args)
    elif len(args.shape) == 3:
        # If 3D array, normalize each slice
        Nxx, Nyy, Nzz = args.shape
        output_args = np.zeros((Nxx, Nyy, Nzz))
        for ii in range(Nzz):
            args[:, :, ii] = normxin(args[:, :, ii])
        return output_args
    else:
        raise ValueError("Input array must be 2D or 3D")    

def test_modulator_info(args):
    global init_cnt
    modulators = []
    
    # First layer
    # input_t = imread('img.png')
    img_size = 264*320
    cnt = init_cnt + 4800
    input_t = MNIST_bin_t[img_size * cnt:img_size * (cnt + 1)].reshape((264, 320))
    pattern = ((input_t.astype(float) - 255) * - 1) / 255
    modulator = modulators_info(pattern, args.input_distance, 0)
    modulators.append(modulator)
    
    pattern_blank = np.ones(pattern.shape)
    pattern = pattern_blank
    for i in range(1, args.layer_number):
        modulator = modulators_info(pattern, args.layer_distance, 0)
        modulators.append(modulator)
    
    return modulators

def expand_modulators(expand_factor, modulator, dev_mask):
    N_yy1, N_xx1 = modulator.shape

    # Expand
    expand_padding_y = int(N_yy1 * (expand_factor - 1) / 2)
    expand_padding_x = int((N_yy1 * expand_factor - N_xx1) / 2)
    modulator = np.pad(modulator, ((expand_padding_y, expand_padding_y), (expand_padding_x, expand_padding_x)), 
                      mode='constant')
    dev_mask = np.pad(dev_mask, 
                      ((expand_padding_y, expand_padding_y), (expand_padding_x, expand_padding_x)), mode='constant')
    
    return modulator, dev_mask

def upsample_modulators(upsample_factor, modulator, dev_mask):
    # Upsample
    new_shape = (int(modulator.shape[0] * upsample_factor), 
                 int(modulator.shape[1] * upsample_factor))
    modulator = resize(modulator, new_shape, order=0)  # nearest neighbor
    dev_mask = resize(dev_mask, new_shape, order=0)
    return modulator, dev_mask

def get_modulator(pattern, dev_flag):
    """
    Define modulation masks here;
    dev_flag - 1: Phase modulation, 0: Amplitude modulation
    modulator: modulation pattern
    dev_mask: dev (e.g., SLM or DMD) modulation window
    """
    
    if dev_flag:
        pattern = pattern * 2 * np.pi
    modulator = pattern
    
    dev_mask = np.ones(modulator.shape)
    N_yy0, N_xx0 = modulator.shape

    # Pad to make it square
    padding_y = int((N_xx0 - N_yy0) / 2)
    modulator = np.pad(modulator, ((padding_y, padding_y), (0, 0)), mode='constant')
    dev_mask = np.pad(dev_mask, ((padding_y, padding_y), (0, 0)), mode='constant')
    
    N_yy, N_xx = modulator.shape
    N_orig = max(N_yy, N_xx)
    # For FFT efficiency and to avoid aliasing, extend N to power of 2 with zero padding
    N = 2 ** int(np.ceil(np.log2(N_orig * 1.5)))  #适当增加N以进行补零，防止混叠
    if N < N_orig:
        N = N_orig  # Ensure N is not less than original pixel count
    
    # Pad input_beam to size N×N
    start_x = int((N - N_xx) / 2)
    end_x = start_x + N_xx
    start_y = int((N - N_yy) / 2)
    end_y = start_y + N_yy
    
    modulator_out = np.zeros((N, N))
    modulator_out[start_y:end_y, start_x:end_x] = modulator
    dev_mask_out = np.zeros((N, N))
    dev_mask_out[start_y:end_y, start_x:end_x] = dev_mask
    
    return modulator_out, dev_mask_out, N

def create_Gauss_beam(pixel_pitch_m, input_Gauss_radius_m, dev_mask):
    """
    Create Gauss input beam based on specified dims and modulation window size
    """
    N_yy, N_xx = dev_mask.shape
    Input_beam = np.zeros((N_yy, N_xx))
    Input_beam_am = gauss_beam_am(N_yy, N_xx, pixel_pitch_m, input_Gauss_radius_m)
    Input_beam_am = np.sqrt(Input_beam_am)
    Input_beam_am = Input_beam_am * dev_mask
    Input_beam = Input_beam_am * np.exp(0j)
    
    return Input_beam

def get_modulated_field(modulator, input_beam):
    """
    Simulate modulation layer, return modulated pattern field
    """
    wave_front = modulator * input_beam

    # # Place original field in center of zero-padded array
    # N_yy, N_xx = modulator.shape
    # start_x = int((N - N_xx) / 2)
    # end_x = start_x + N_xx
    # start_y = int((N - N_yy) / 2)
    # end_y = start_y + N_yy

    # U0_padded = np.zeros((N, N), dtype=complex)
    # U0_padded[start_y:end_y, start_x:end_x] = wave_front

    # U0 = U0_padded  # Use zero-padded field for calculation
    
    return wave_front

def create_modulator(modulator_info):
    modulator, dev_mask, N = get_modulator(modulator_info.pattern, modulator_info.dev_flag)
    
    # Expand and upsample the modulator for higher resolution if necessary
    modulator, dev_mask = expand_modulators(args.expand_factor, modulator, dev_mask)
    
    modulator, dev_mask = upsample_modulators(args.upsample_factor, modulator, dev_mask)
    args.pixel_pitch = args.pixel_pitch / args.upsample_factor
    
    if modulator_info.dev_flag:
        modulator = np.exp(1j * modulator)
    
    return modulator, dev_mask, N

def crop(target, dx, dy, N):
    simulation_width_m = args.simulation_width * 1e-3                         # m
    
    N_cut = int(np.ceil(simulation_width_m/dx/2)*2)
    N_cut_half = int(N_cut/2)
    N_center = int(np.floor((N+1)/2))

    output_cut = target[N_center-N_cut_half:N_center+N_cut_half, 
                                         N_center-N_cut_half:N_center+N_cut_half]

    output_intensity_norm_cut = output_cut * np.conj(output_cut)
    output_intensity_norm_cut = normxin(output_intensity_norm_cut)
    output_intensity_norm_cut = np.nan_to_num(output_intensity_norm_cut)
    
    return output_intensity_norm_cut
   
def spatial_propagation(input_pattern, pixel_pitch_m, N, k_vector, d):
    # Extended physical dimensions (for FFT calculation)
    dx = pixel_pitch_m  # Initial plane pixel spacing (m)
    dy = pixel_pitch_m  # Initial plane pixel spacing (m)
    Lx_padded = dx * N
    Ly_padded = dy * N
    
    # Calculate frequency domain coordinates
    dfx = 1 / Lx_padded  # Frequency domain sampling interval
    dfy = 1 / Ly_padded
    fx = np.arange(-N/2, N/2) * dfx
    fy = np.arange(-N/2, N/2) * dfy
    Fx, Fy = np.meshgrid(fx, fy)

    # Fourier transform of initial field to get angular spectrum
    A0 = fftshift(fft2(input_pattern))

    # Wave number k_vector
    # k_vector = 2 * pi / lambda_m

    # Transfer function H(fx, fy, z)
    # kz = sqrt(k_vector^2 - (2*pi*fx)^2 - (2*pi*fy)^2)
    # Note: When kx^2 + ky^2 > k_vector^2, kz is imaginary, corresponding to evanescent waves
    kz_squared = k_vector**2 - (2*np.pi*Fx)**2 - (2*np.pi*Fy)**2

    # Handle evanescent waves
    kz = np.sqrt(kz_squared.astype(complex))  # Use complex to handle negative values

    simulation_d_m = d * 1e-3

    H = np.exp(1j * kz * simulation_d_m)
    Az = A0 * H
    prop_pattern = ifft2(ifftshift(Az))
    
    return prop_pattern

def save_imgs(simulation_output):
    global init_cnt
    # Create directory for results
    dir_temp = 'ASM_simulation_results'
    os.makedirs(dir_temp, exist_ok=True)

    dir_temp2 = os.path.join(dir_temp, 'norm2D')
    os.makedirs(dir_temp2, exist_ok=True)

    # Save results
    output_img = norm2D(simulation_output.astype(float))
    img_dir = 'norm2D_' + str(init_cnt) + '.tiff' #+ datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.tiff'
    init_cnt = init_cnt + 1
    plt.imsave(os.path.join(dir_temp2, img_dir), 
               output_img, 
               cmap='gray')
    if init_cnt % 10 == 1:
        plt.imshow(output_img)
        plt.title('Output result')
        plt.show()
    # print(f"Results saved to {dir_temp}")

def simu_main(args):
    # Get the modulators
    if args.layer_number < 1:
        return False, 'Layer number <1!'
    modulators = test_modulator_info(args)
    
    # Create first layer modulator
    modulator, dev_mask, N = create_modulator(modulators[0])
    
    if args.see_patterns:
        plt.imshow(modulator)
        plt.title('Input layer pattern')
        plt.show()
    
    # Create input beam
    pixel_pitch_m = args.pixel_pitch * 1e-6
    input_Gauss_radius_m = args.input_Gauss_radius * 1e-3  
    input_beam = create_Gauss_beam(pixel_pitch_m, input_Gauss_radius_m, dev_mask)
    
    if args.see_patterns:
        plt.figure(1)
        plt.imshow(np.abs(input_beam), cmap='gray')
        plt.title('Input Beam')
        plt.show()
    
    # For ininital input, propagation simu is not necessarity required due to Gaussian beam
    # However, prop simu is required for following stages of reflections
    lambda_m = args.input_lambda * 1e-9                                       # m
    lambda_media = lambda_m / args.refractive_index                           # wavelength in the media
    k_vector = 2 * np.pi / lambda_media  
    simulation_input = spatial_propagation(input_beam, pixel_pitch_m, N, k_vector, modulators[0].d_layer)
    
    # Compute modulated wave field for first layer
    # print("Enter layer1!")
    U0 = get_modulated_field(modulator, simulation_input)
    
    # Compute modulated wave field for following layers
    for i in range(1, args.layer_number):
        # print("Enter layer" + str(i + 1) + "!")
        
        # Input beam to next layer
        simulation_next = spatial_propagation(U0, pixel_pitch_m, N, k_vector, modulators[i].d_layer)
        
        ## Update modulator here is necessary
        modulator, _, _ = create_modulator(modulators[i])
        
        if args.see_patterns:
            plt.imshow(modulator)
            plt.title('Layer ' + str(i) + ' pattern')
            plt.show()
        
        # Compute modulated wave field for next layer
        U0 = get_modulated_field(modulator, simulation_next)
    
    # Propagation simulation via angular spectrum
    simulation_output = spatial_propagation(U0, pixel_pitch_m, N, k_vector, args.output_distance)
    # Crop results to observation window size
    output = crop(simulation_output, pixel_pitch_m, pixel_pitch_m, N)
    
    # Save results
    save_imgs(output)
    
    return True, 0

if __name__ == "__main__":
    args = parse_args()
    for i in range(0, 1200):
        ret, result = simu_main(args)
        if not ret:
            print(f"Error: {result}")