from torch.autograd.function  import Function, InplaceFunction
import neg_dataset_construct as neg_data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from threading import Thread
import numpy as np
import torch.nn as nn
import cv2 as cv
import torch
import math
import sys
import time

###################
# get_neg_label:  get negative dataset for FF algorithm
###################
def get_neg_label(data_size, labels, is_dmd):
    return neg_data.construct_neg_set(data_size, labels, is_dmd)

###################
# apply_overall_integer_shift:  Pollut input image with pixel misalignment;
#                               drifted pixel number is image-wise Gaussian distributed 
###################
max_shift_default = 10
O_shift_factor = 12.8
def apply_overall_integer_shift(
    image: np.ndarray,
    image_num: int = 0,
    dim_x: int = 8,
    dim_y: int = 10,
    sigma: float = 0.25,
    max_shift: int = 2,
    seed: int | None = None,
    fill_value: int = 0,
) -> tuple[np.ndarray, tuple[int, int]]:
   
    if image.ndim != 2:
        raise ValueError("Unsupported image shape: must be 2D")

    h = dim_x
    w = dim_y
    
    for i in range(0, image_num):
        rng = np.random.RandomState(seed)
        dx = float(rng.normal(loc=0.0, scale=sigma))
        dy = float(rng.normal(loc=0.0, scale=sigma))
        dx = np.clip(dx, -abs(max_shift), abs(max_shift))
        dy = np.clip(dy, -abs(max_shift), abs(max_shift))
        dx_int = int(np.rint(dx))
        dy_int = int(np.rint(dy))
        
        image_t = image[i].reshape((dim_x, dim_y))
        out = np.full((h, w), int(fill_value), dtype=image_t.dtype)

        # Destination region in output
        dst_y0 = max(0, dy_int)
        dst_y1 = min(h, h + dy_int)
        dst_x0 = max(0, dx_int)
        dst_x1 = min(w, w + dx_int)
    
        # Corresponding source region in input
        src_y0 = dst_y0 - dy_int
        src_y1 = dst_y1 - dy_int
        src_x0 = dst_x0 - dx_int
        src_x1 = dst_x1 - dx_int
    
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = image_t[src_y0:src_y1, src_x0:src_x1]
        
        out = np.ascontiguousarray(out)
        image[i] = out.reshape((-1, ))

    return image

###################
# get_input:  get input images in DMD buffer format
###################
# for test
see_img_id = 1
width_in = 264
length_in = 320
input_size = width_in * length_in 
def get_input(start, end, input_data, is_pix_err):
    input_imgs = input_data[start * input_size : end * input_size,]
    
    if is_pix_err:
        img_num = end - start
        input_imgs = apply_overall_integer_shift(input_imgs.reshape((img_num, -1)),\
                                                 image_num=img_num, dim_x=width_in, dim_y=length_in,\
                                                     max_shift=max_shift_default*O_shift_factor)
    
    # 如果不是C连续的内存，必须强制转换
    if not input_imgs.flags['C_CONTIGUOUS']:
        input_imgs = np.ascontiguous(input_imgs, dtype=input_imgs.dtype)
    return input_imgs

###################
# Binarize:  transform a tensor into binerzied version, and defines its backpropagation behaviour
###################
class Binarize(InplaceFunction):

    def forward(ctx,input,quant_mode='det',allow_scale=False,inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()      

        scale = output.abs().max() if allow_scale else 1

        if quant_mode=='det':
            return output.div(scale).sign().mul(scale)
        else:
            return output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1).mul(scale)
        
    def backward(ctx,grad_output):
        #STE 
        grad_input=grad_output
        return grad_input,None,None,None

def binarized(input,quant_mode='det'):
      return Binarize.apply(input,quant_mode)      

###################
# weight_to2d:  transform a tensor weight matrix (layer_num * (input_height * input_width))
#               into DMD img 1D buffer format
###################
# overall layer pixel size on DMD (8 * 20 * 2) * (10 * 21 * 2) = 320 * 420
# how many weight pixel for each weight height
dim_1 = 8
# how many weight for each col in layer
dim_2 = 20
# how many weight pixel for each weight width
dim_3 = 10
# how many weight for each row in layer
dim_4 = 22
# The size of a 'weight pixel' for DMD pixel
base_w = 2
base_h = 2

def weight_to2d(weight, dim_switch, dim_factor):
    weight = weight.cpu()
    weight_np = weight.detach().numpy()
    weight_np = np.where(weight_np < 0, 0, 1)
    pix_h = int(dim_1 * dim_factor)
    pix_w = int(dim_3 * dim_factor)
    weight_h = int(dim_2 / dim_factor)
    weight_w = int(dim_4 / dim_factor)
    # change from 420 * 80 to 80 * 420 if requsted
    if dim_switch:
        pix_h = dim_3
        pix_w = dim_1
        weight_h = dim_4
        weight_w = dim_2
    
    weight_pix_num = base_w * base_h
    weight_num = weight_h * weight_w
    dmd_pix_w = pix_w * base_w
    dmd_w = dmd_pix_w * weight_w
    pix_num = pix_h * pix_w
    
    # each DMD pixel is an uint8, (8 * 20 * 2) * (10 * 21 * 2) = 320 * 420
    weight_buf = np.zeros((1, weight_num * pix_num * weight_pix_num)).astype('uint8')
    # expend weight into DMD size 420 * 80 -> 420 * (8 * 2 * 10 * 2)
    weight_layer = cv.resize(weight_np.astype('uint8'), (pix_num * base_w, weight_num * base_h))
    
    for i in range(0, weight_num):
        for j in range(0, pix_h):
            for k in range(0, base_h):
                pos = (i % weight_w) * dmd_pix_w + (i // weight_w) * pix_num \
                    * weight_w * weight_pix_num + j * dmd_w * base_h + k * dmd_w
                weight_buf[0][pos: pos + dmd_pix_w] \
                    = weight_layer[i * base_h + k][j * dmd_pix_w : (j + 1) * dmd_pix_w]
    
    return weight_buf

###################
# get_weights:  returns a dmd-form weight buffer contains all weights in it
###################
max_layer_num = 3
def get_weights(network, DMD_layer_dim, layer_num, dim_factor):
    # get binarized weight layers
    w_1 = network.layer1.weight.clone()
    w_2 = network.layer2.weight.clone()
    w_3 = network.layer3.weight.clone()
    w_1_b = binarized(w_1)
    w_2_b = binarized(w_2)
    w_3_b = binarized(w_3)
    
    
    for w_id in range(0, layer_num):
        locals()["w_" + str(w_id + 1) + "_b"] \
            = binarized(locals()["w_" + str(w_id + 1)])
    for w_id in range(layer_num, max_layer_num):
        locals()["w_" + str(w_id + 1) + "_b"].fill(0x0)
    
    # set weight layers in DMD buffer form
    pix_dim = DMD_layer_dim * base_w * base_h
    DMD_layers = np.zeros((1, max_layer_num * pix_dim)).astype('uint8')
    for w_id in range(0, layer_num):
        DMD_layers[0, w_id * pix_dim : (w_id + 1) * pix_dim] \
            = weight_to2d(locals()["w_" + str(w_id + 1) + "_b"], w_id % 2, dim_factor)
    DMD_layers = np.where(DMD_layers > 0, 0xFF, 0)
    DMD_layers = DMD_layers.astype('uint8')
    if not DMD_layers.flags['C_CONTIGUOUS']:
        DMD_layers = np.ascontiguous(DMD_layers, dtype='uint8')
    
    return DMD_layers

###################
# ccd_img_collect:  collect ccd-captured images from given img_path;
#                   confirm img_path_t before use
###################
see_img_id = 1
ccd_img_len = 540
ccd_img_wid = 400
img_path_t = './images/Acquisition-'
def ccd_img_collect(MNIST_loop_num, cap_img_num, padding_num):
    per_loop = int(cap_img_num / MNIST_loop_num)
    ccd_imgs = np.zeros((cap_img_num, ccd_img_len, ccd_img_wid))
    for i in range(0, MNIST_loop_num) :
        for j in range(0, per_loop + padding_num) :
            if j < padding_num:
                continue
            curr_id = i * per_loop + j - padding_num
            img_path = img_path_t + str(i) + '-' + str(j) + '.jpg'
            img_t = mpimg.imread(img_path)
            img_n = img_t[:, 20:370]
            ccd_imgs[curr_id] = cv.resize(img_n, (ccd_img_wid, ccd_img_len))
            
    ccd_img_t = ccd_imgs[see_img_id - 1]
    ccd_imgs = (ccd_imgs - np.min(ccd_imgs)) / (np.max(ccd_imgs) - np.min(ccd_imgs))
    ccd_img_out = np.reshape(ccd_imgs, (cap_img_num, ccd_img_len*ccd_img_wid))
    
    return ccd_img_out, ccd_img_t

###################
# Define thread funcs with return valus
###################
class OnnThread(Thread):
    def __init__(self, func, args):
        super(OnnThread, self).__init__()
        self.func = func
        self.args = args
        
    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

#########
# Power-Law / Gamma activation
#########
class GammaActivation(nn.Module):
    def __init__(self, gamma):
        super(GammaActivation, self).__init__()
        self.gamma = gamma
        
    def forward(self, x):
        return torch.pow(x, self.gamma)

###################
# System struct defines
###################
class datasets:
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

class net_data:
    def __init__(self, test_size, epoches, dataset_name):
        self.dataset = dataset_name
        self.max_acc = 0
        self.max_acc_loss = 0
        self.min_loss = 100
        self.min_loss_acc = 0
        self.max_acc_pred = np.zeros(test_size)
        self.min_loss_pred = np.zeros(test_size)
        self.acc_evo = np.zeros((epoches))
        self.loss_evo = np.zeros((epoches))
        self.max_acc_output = np.zeros((test_size, ccd_img_len*ccd_img_wid)).astype('uint8')
        self.vsensor_out = None

class NN_components:
    def __init__(self, fcn, lrn, lrn_model, loss_func, optimizer,\
                                         loss_func_lr, optimizer_lr):
        self.fcn = fcn
        self.lrn = lrn
        self.lrn_model = lrn_model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.loss_func_lr = loss_func_lr
        self.optimizer_lr = optimizer_lr
        self.loss = 0

#########
# crossentropy
#########
regularization_rate=0.0004
def crossentropy(y_hat, y_true):
    y_hat[y_hat==0] = 0.00000001   #避免出现log0
    y_hat[y_hat==1] = 1.00000001   #避免出现log0
    loss = -1*np.sum(y_true*np.log(y_hat) + (1-y_true)*np.log(1-y_hat))/y_hat.shape[0]
    dz_hat = -(y_true - y_hat)
    
    loss_w = 0
    # # L2 regulation
    # for w in weight:
    #     loss_w += np.sum(np.power(w, 2))
    # loss_w = loss_w*regularization_rate/y_hat.shape[0]
    loss = loss + loss_w
    return {"loss": loss.cuda(), "dz_hat": dz_hat}

def __optimizer_f(self, optimizer, index,  global_step=None):
    self.m_w[index] = self.dw[index]
    self.m_b[index] = self.db[index]
    m_w, m_b, V_w, V_b = self.m_w[index], self.m_b[index], self.V_w[index], self.V_b[index]
    # 下降梯度yita = lr * mt / sqrt(Vt)
    yita_w = self.learn_rate * (m_w / np.sqrt(V_w))
    yita_b = self.learn_rate * (m_b / np.sqrt(V_b))
    return yita_w, yita_b


#########
# functions for test
#########
# DMD/CCD controling dll
sys.path.append("./DMD_dll")
import DMD_dll

# A test 3-layer FC
class Lr(nn.Module):
    def __init__(self):
        super(Lr,self).__init__()
        self.layer1 = nn.Linear(in_features=80, out_features=440)
        self.layer2 = nn.Linear(in_features=440, out_features=80)
        self.layer3 = nn.Linear(in_features=80, out_features=440)
    # 定义前向传播过程
    def forward(self, x):
        out = self.linear(x)
        return out

# This is a test function for 
def load_img_test():
    # get 50 input images
    input_imgs = get_input(0, 50)
    
    # construct test weights
    weight_t = np.zeros((440, 80))
    weight2_t = np.zeros((80, 440))
    weight_t[0 : 220, :] = 255
    weight2_t[0 : 40, :] = 255
    
    weight_layer1 = torch.from_numpy(weight_t)
    weight_layer2 = torch.from_numpy(weight2_t)
    
    test_lr = Lr()
    test_lr.layer1.weight = nn.Parameter(weight_layer1)
    test_lr.layer2.weight = nn.Parameter(weight_layer2)
    test_lr.layer3.weight = nn.Parameter(weight_layer1)
    
    input_weight = get_weights(test_lr, 440 * 80)
    
    # import DMD funcs here, from DMD_dll.py
    with DMD_dll.dmd_funcs() as dmd:
        ret = dmd.test_img(input_imgs, input_weight, 10)
    
    print("load_img_test ret = ", ret)
    
    
# change test function name and run this script to test individual misc functions
if __name__ == "__main__":
    load_img_test()

