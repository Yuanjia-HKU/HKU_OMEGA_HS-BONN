from torchvision import datasets, transforms
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.optim as optim
import ONN_misc as misc
import scipy.io as sio
import torch.nn as nn
import vsensor_model
import numpy as np
import ctypes
import torch
import math
import time
import sys
import os

# os.chdir("D:/Yuanjia/ONN/python")

# Use GPU
torch.cuda.set_device(0)

# DMD/CCD controling dll
sys.path.append("./DMD_dll")
import DMD_dll
from DMD_dll import dmd_funcs
dmd_t = dmd_funcs()

# Task scheduler
sys.path.append("./Engine_scheduler")
import Engine_scheduler

###########################
# Set system instructions
###########################
is_dmd = 1
# Select dataset, method 1 has priority
## 1. Directly point to dataset: (set to 'None' if not used)
## Support MNIST digit, MNIST fashion, Vowel, ASL
dataset = None
## 2. Select dataset by dataset idx, see dataset_table
dataset_id = 0
# Training epoch num
epoches = 200
# Learning rate of ONN
learning_rate = 0.04
# Learning rate of decision layer
learning_rate_lr = 0.003
# Hidden layer num
test_layer_num = 3
# Parallel tasks
is_dual = 0 and is_dmd # Currently only support same datasets for comparison, i.e., same labels
# Use z-acc sensor
is_vsensor = 0
# Try fully-precision networks
is_FPFC = 0 and (not(is_dmd))
FC_activation = misc.GammaActivation(gamma = 2.0)
# Print run times
show_time = 0
# Add pixel misalignment noise
is_pix_err = 0
max_shift_num = misc.max_shift_default
E_shift_factor = 2.5
# 为了每次的实验结果一致
# torch.manual_seed(1)
###########################
# get dataset contents
###########################
dataset_name, dataset_dmd = 'NA', None
data_set_size, train_set_size, batch_size = 0, 0, 0
dim_factor = 0
output_class = 10
# MNIST
def Mnist_digit():
    global dataset_name, dataset, dataset_dmd, data_set_size, train_set_size,\
        batch_size, dim_factor
    dataset_name = "MNIST"
    dataset = MNIST_rep_b
    dataset_dmd = misc.datasets(MNIST_bin_t, "MNIST_bin_t")
    train_set_ratio = 0.8
    data_set_size = 6000
    train_set_size = int(data_set_size * train_set_ratio)
    batch_size_step = 3000
    batch_size = 2 ** (int(math.ceil(data_set_size / batch_size_step)) - 1) * 50
    dim_factor = 1
# Vowel
def Vowel_voice():
    global dataset_name, dataset, dataset_dmd, data_set_size, train_set_size,\
        batch_size, dim_factor
    dataset_name = "vowel"
    dataset = vowel_rep_b
    dataset_dmd = misc.datasets(vowel_bin_t, "vowel_bin_t")
    data_set_size = 259
    train_set_size = 200
    batch_size = 50
    dim_factor = 2
# MNIST_Fashion
def Mnist_fashion():
    global dataset_name, dataset, dataset_dmd, data_set_size, train_set_size,\
        batch_size, dim_factor
    dataset_name = "fashion"
    dataset = fashion_rep_b
    dataset_dmd = misc.datasets(fashion_bin_t, "fashion_bin_t")
    train_set_ratio = 0.8
    data_set_size = 6000
    train_set_size = int(data_set_size * train_set_ratio)
    batch_size_step = 2000
    batch_size = 2 ** (int(math.ceil(data_set_size / batch_size_step)) - 1) * 50
    dim_factor = 2
# MNIST added noise
def Mnist_noise():
    global dataset_name, dataset, dataset_dmd, data_set_size, train_set_size,\
        batch_size, dim_factor
    dataset_name = "MNIST_noise"
    dataset = noise_rep_b
    dataset_dmd = misc.datasets(noise_bin_t, "noise_bin_t")
    train_set_ratio = 0.8
    data_set_size = 6000
    train_set_size = int(data_set_size * train_set_ratio)
    batch_size_step = 3000
    batch_size = 2 ** (int(math.ceil(data_set_size / batch_size_step)) - 1) * 50
    dim_factor = 1
# ASL gesture for digits
def ASL_digit_gesture():
    global dataset_name, dataset, dataset_dmd, data_set_size, train_set_size, \
        batch_size, dim_factor
    dataset_name = "ASL_gesture"
    dataset = ASL_rep_b
    dataset_dmd = misc.datasets(ASL_bin_t, "ASL_bin_t")
    data_set_size = 700
    train_set_size = 550
    batch_size = 50
    dim_factor = 2
# ASL gesture for letters:
def ASL_letter_gesture():
    global dataset_name, dataset, dataset_dmd, data_set_size, train_set_size,\
        batch_size, dim_factor, output_class
    dataset_name = "ASL_letter"
    dataset = ASL_letter_rep_b
    dataset_dmd = misc.datasets(ASL_letter_bin_t, "ASL_letter_bin_t")
    data_set_size = 1800
    train_set_size = 1400
    batch_size = 50
    dim_factor = 2
    output_class = 26
# Cifar10
def Cifar10():
    global dataset_name, dataset, dataset_dmd, data_set_size, train_set_size,\
        batch_size, dim_factor
    dataset_name = "Cifar10"
    dataset = Cifar10_rep_b
    dataset_dmd = misc.datasets(Cifar10_k6o_bin_t, "Cifar10_bin_t")
    train_set_ratio = 0.8
    data_set_size = 6000
    train_set_size = int(data_set_size * train_set_ratio)
    batch_size_step = 2000
    batch_size = 2 ** (int(math.ceil(data_set_size / batch_size_step)) - 1) * 50
    dim_factor = 2
# Imagenette
def Imagenette():
    global dataset_name, dataset, dataset_dmd, data_set_size, train_set_size,\
        batch_size, dim_factor
    dataset_name = "Imagenette"
    dataset = Imagenette_rep_b
    dataset_dmd = misc.datasets(Imagenette_bin_t, "Imagenette_bin_t")
    train_set_ratio = 0.8
    data_set_size = 6000
    train_set_size = int(data_set_size * train_set_ratio)
    batch_size_step = 2000
    batch_size = 2 ** (int(math.ceil(data_set_size / batch_size_step)) - 1) * 50
    dim_factor = 1

dataset_table = {
    0: Mnist_digit,
    1: Vowel_voice,
    2: Mnist_fashion,
    3: ASL_digit_gesture,
    4: ASL_letter_gesture,
    5: Cifar10,
    6: Imagenette,
    7: Mnist_noise,
    }

# Get dataset information by dataset idx
if dataset is None:
    if dataset_id >= len(dataset_table) or dataset_id < 0:
        print('Dataset_id invalid!')
        sys.exit(1)
# Get dataset information with engine scheduler
else:
    dataset_t = dataset.reshape((-1, misc.width_in, misc.length_in))
    dataset_id = Engine_scheduler.Engine_schedule(dataset_t[0 : Engine_scheduler.sample_data_num])
    
    # Distinguish ASL digits and ASL letters
    if dataset_id == 3 and len(dataset_t) > 700:
        dataset_id = 4

get_dataset = dataset_table.get(dataset_id)
get_dataset()
print('Get', dataset_name, 'tasks!')

###########################
# Parallel dataset setting
###########################
datasets_dmd = []
datasets_dmd.append(dataset_dmd)
if is_dual:
    dataset2_dmd = misc.datasets(fashion_bin_t, "fashion_bin_t")
    datasets_dmd.append(dataset2_dmd)

###########################
# network/DMD dimention setting
###########################
Ex_in = 8
Ey_in = 10
in_dim = int( (Ex_in * dim_factor) * (Ey_in * dim_factor))
n_hidden_1 = int ( (22 / dim_factor) * (20 / dim_factor) )
n_hidden_2 = int ( (22 / dim_factor) * (20 / dim_factor) )
out_dim = int ( (22 / dim_factor) * (20 / dim_factor) )
out_dim_dmd = 400 * 540
if is_dmd:
    in_dim_lr = out_dim_dmd
else:
    in_dim_lr = out_dim
out_dim_lr = output_class
DMD_layer_dim = in_dim * n_hidden_1
DMD_layer_num = 3
DMD_padding_num = 5
DMD_buffer_size = 200

###########################
# Run time stastics
###########################
epoch_start_time = 0
IMG_grip_time = 0
DMD_run_time = 0
update_time = 0
model_time = 0
test_time = 0
def print_time():
    global IMG_grip_time, DMD_run_time, update_time, model_time, epoch_start_time, test_time
    print("Epoch sum time:", time.time() - epoch_start_time, "s")
    print("Epoch DMD_run_time:", DMD_run_time, "s")
    print("Epoch Model_run_time:", model_time, "s")
    print("Epoch Update_run_time:", update_time, "s")
    print("Epoch Collect_img_time:", IMG_grip_time, "s")
    print("Epoch test_time:", time.time() - test_time, "s")
    IMG_grip_time = 0
    DMD_run_time = 0
    update_time = 0
    model_time = 0
    test_time = 0

###########################
# Viberation sensor init
###########################
vsensor_out = []
is_task_finish = 0
def vsensor_thread_func(device, is_sensor):
    if is_sensor >= 0:
        device.startLoopRead()
        time.sleep(0.5)
        
        while True:
            az_val = device.get("54")
            az_val = ctypes.c_int16(az_val).value
            vsensor_out.append(az_val * 16 / 32768)
            time.sleep(0.2)
            if is_task_finish:
                break
    device.stopLoopRead()
    time.sleep(0.5)
    device.closeDevice()
    return

###########################
# Load datasets
###########################
filename = './labels/' + dataset_name + '_label.mat'
mat_contents = sio.loadmat(filename)
label_data01 = mat_contents['label']
label_data02 = np.reshape(label_data01[0 : data_set_size], (data_set_size, -1))
train_label = label_data02[0 : train_set_size, :].squeeze(1)
test_label = label_data01[train_set_size : data_set_size, :].squeeze(1)
test_set_size = data_set_size - train_set_size
class DMD_imgs(Data.Dataset):
    def __init__(self, dataset):
        train_data = (np.array(dataset[0 : train_set_size, :, :], dtype='f')).reshape((train_set_size, in_dim))
        self.data = torch.tensor(train_data)
        self.label = torch.LongTensor(train_label)

    def __getitem__(self,index):
        return (self.data[index]).cuda(),(self.label[index]).cuda()
    
    def __len__(self):
        return len(self.data)
    
class test_imgs(Data.Dataset):
    def __init__(self, dataset):
        test_data = (np.array(dataset[train_set_size : data_set_size, :, :], dtype='f').reshape((test_set_size, in_dim)))
        self.data = torch.tensor(test_data)
        self.label = torch.LongTensor(test_label)

    def __getitem__(self,index):
        return (self.data[index]),(self.label[index])

    def __len__(self):
        return len(self.data)

# 将训练数据装入Loader中
train_loader = Data.DataLoader(dataset=DMD_imgs(dataset), batch_size=batch_size, shuffle = False)
test_set = test_imgs(dataset)
print(test_set.data.size())
test_x = test_set.data.type(torch.FloatTensor)
test_y = test_set.label

###########################
# Network structure design
###########################
net_dims = np.zeros((misc.max_layer_num))
net_dims[0] = in_dim
net_dims[1] = n_hidden_1
net_dims[2] = n_hidden_2
# define binarized FC
class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != in_dim:
            input_b = misc.binarized(input)
        else:
            input_b = input
        weight_b = misc.binarized(self.weight)
        
        out = nn.functional.linear(input_b,weight_b)

        return out

class Fc(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Fc,self).__init__()
        # fully-precesion layers
        if is_FPFC:
            self.layer1 = nn.Linear(in_dim,n_hidden_1,bias=False)
            self.layer2 = nn.Linear(n_hidden_1,n_hidden_2,bias=False)
            self.layer3 = nn.Linear(n_hidden_2,out_dim,bias=False)
            self.activation = FC_activation
        else:
            # BNN
            self.layer1 = BinarizeLinear(in_dim,n_hidden_1,bias=False)
            self.layer2 = BinarizeLinear(n_hidden_1,n_hidden_2,bias=False)
            self.layer3 = BinarizeLinear(n_hidden_2,out_dim,bias=False)
        
    def forward(self,x):
        # is_FPFC: conventional forward (use fully-precision network, as it should be)
        # else:    simulated forward (extract "forward" out)
        layer_1 = self.layer1
        layer_2 = self.layer2
        layer_3 = self.layer3
        layer_n = locals()
        hidden_out = x
        for layer_id in range(0, test_layer_num):
            layer_t = layer_n["layer_" + str(layer_id + 1)]
            weight = layer_t.weight.clone()
            if not(is_FPFC):
                weight = misc.binarized(weight)
            hidden_out = nn.functional.linear(hidden_out, weight)
            if is_FPFC and self.activation != None:
                hidden_out = self.activation(hidden_out)
            if layer_id < (test_layer_num - 1):
                hidden_out = normalize(hidden_out)
        
        out = hidden_out
        
        return out

# define physical DMD forward behaviour, in replacement of above digital network forward
def dmd_batch_foward(network, in_start, in_end, dataset):
    global IMG_grip_time, DMD_run_time, test_layer_num
    
    input_imgs = misc.get_input(in_start, in_end, dataset, is_pix_err)
    weights = misc.get_weights(network, DMD_layer_dim, test_layer_num, dim_factor)
    image_num = in_end - in_start
    # load to dmd and run
    t_start = time.time()
    # import DMD funcs here, from DMD_dll.py
    ret = dmd_t.run_dmd(input_imgs, weights, image_num)
    if ret != 0:
        print("WARN: dmd_run ret = ", ret)
    DMD_run_time = DMD_run_time + time.time() - t_start
    
    # get result from CCD (may need modify)
    t_start = time.time()
    out, out_t = misc.ccd_img_collect(int(math.ceil(image_num / DMD_buffer_size)), image_num, DMD_padding_num)
    IMG_grip_time = IMG_grip_time + time.time() - t_start
    out = torch.tensor(out)
    
    return out.type(torch.FloatTensor), out_t

# define a simple decision layer
class Lr(nn.Module):
    def __init__(self, in_dim_lr, out_dim_lr):
        super(Lr,self).__init__()
        self.linear = nn.Linear(in_features=in_dim_lr, out_features=out_dim_lr)
        self.activate = nn.ReLU()
    # 定义前向传播过程
    def forward(self, x):
        out = self.activate(x)
        out = self.linear(out)
        return out

# make simulated model running back-stage to save time
def model_thread_func(input_x, fc, lr):
    global model_time
    t_start = time.time()
    
    input_t = input_x.clone()
    if is_pix_err:
        input_t = input_t.cpu()
        input_t = misc.apply_overall_integer_shift(input_t.numpy(), image_num=input_t.size()[0], \
                                                   dim_x=Ex_in*dim_factor, dim_y=Ey_in*dim_factor, \
                                                       max_shift=int(max_shift_num/E_shift_factor))
        input_t = torch.from_numpy(input_t)
        input_t = input_t.cuda()
        
    output_fc = fc(input_t)
    output_model = output_fc.view(output_fc.size(0), -1)
    output_model = lr(output_model)
    
    model_time = model_time + time.time() - t_start
    return output_model

    
outputs = []
for i in range(0, len(datasets_dmd)):
    output = misc.net_data((test_y.data.numpy()).size, epoches, datasets_dmd[i].name)
    outputs.append(output)
###########################
# ONN system main
###########################    
def main():
    global IMG_grip_time, DMD_run_time, update_time, model_time,\
            epoch_start_time, test_time
    dmd_dataset_num = len(datasets_dmd)
    
    # init NNs for each parallel-running datasets
    NNs = []
    for i in range(0, dmd_dataset_num):
        fcn = Fc(in_dim,n_hidden_1,n_hidden_2,out_dim)
        fcn.cuda()
        
        lrn = Lr(in_dim_lr, out_dim_lr)
        lrn.cuda()
        if is_dmd:
            lrn_model = Lr(out_dim, out_dim_lr)
            lrn_model.cuda()
            lrn_t = lrn_model
        else:
            lrn_t = lrn
    
        # 定义优化器和损失函数
        loss_func_lr = nn.CrossEntropyLoss()
        optimizer_lr = torch.optim.SGD(lrn.parameters(), lr = learning_rate_lr)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(fcn.parameters(), lr = learning_rate)
        
        # pack NN[i] components
        NN_component = misc.NN_components(fcn, lrn, lrn_t, loss_function, optimizer, loss_func_lr, optimizer_lr)
        NNs.append(NN_component)
    
    # init viberation sensor
    if is_vsensor:
        s_device = vsensor_model.DeviceModel("测试设备", "COM3", 9600, 0x50)
        is_sensor = s_device.openDevice()
        vsensor_thread = misc.OnnThread(vsensor_thread_func, args = (s_device, is_sensor))
        vsensor_thread.start()
    
    if is_dmd:
        dmd_t.dmd_init()
    
    for epoch in range(epoches):
        print("-Enter epoch{}-".format(epoch))
        data_cnt = 0
        epoch_start_time = time.time()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # use simulated network for each paralleled dataset
            model_threads = []
            for i in range(0, dmd_dataset_num):
                model_thread = misc.OnnThread(model_thread_func, args = (batch_x, NNs[i].fcn, NNs[i].lrn_model))
                model_thread.start()
                model_threads.append(model_thread)
              
            # use dmd to forward the network for each paralleled dataset
            if is_dmd:
                train_start = data_cnt
                train_end = data_cnt + batch_size
                outputs_dmd = []
                for i in range(0, dmd_dataset_num):
                    output_dmd, out_img_t = dmd_batch_foward(NNs[i].fcn, train_start, train_end, datasets_dmd[i].dataset)
                    # print(batch_y.data.numpy())
                    output_dmd = output_dmd.cuda()
                    output_dmd = NNs[i].lrn(output_dmd)
                    outputs_dmd.append(output_dmd)
                    time.sleep(1.5)
                    if step % 10 == 0:
                        plt.imshow(out_img_t)
                        plt.show()
            
            # use model to forward the network for each paralleled dataset
            for i in range(0, dmd_dataset_num):
                model_threads[i].join()
                output_model = model_threads[i].get_result()
            
                t_start = time.time()
                with torch.autograd.set_detect_anomaly(True):
                    output_fcn = output_model.clone()
                    if is_dmd:
                        output_t = outputs_dmd[i].clone().detach()
                        output_fcn.data = torch.tensor(output_t)
                        output_lrn = outputs_dmd[i].clone()
                    else:
                        output_lrn = output_model.clone()
                        
                    # calculate and update 3-layer fc
                    NNs[i].loss = NNs[i].loss_func(output_fcn, batch_y)
                    NNs[i].optimizer.zero_grad()
                    # loss_fcn = loss_fcn.detach_().requires_grad_(True)
                    NNs[i].loss.backward(retain_graph=True)
                    NNs[i].optimizer.step()
                    
                    # calculate and update decision layer
                    loss_lr = NNs[i].loss_func_lr(output_lrn, batch_y)
                    NNs[i].optimizer_lr.zero_grad()
                    loss_lr.backward()
                    NNs[i].optimizer_lr.step()
                update_time = update_time + time.time() - t_start
        
            # test acuracy every epoch
            if (step + 1) % (train_set_size / batch_size) == 0:
                test_time = time.time()
                for i in range(0, dmd_dataset_num):
                    if  is_dmd:
                        test_start = train_set_size
                        test_end = data_set_size
                        test_output, out_img_t = dmd_batch_foward(NNs[i].fcn, test_start, test_end, datasets_dmd[i].dataset)
                        test_output = test_output.cuda()
                        plt.imshow(out_img_t)
                        plt.title("Epoch %s" % epoch + " dataset%s output" % i)
                        plt.show()
                        
                    else:
                        test_output = NNs[i].fcn(test_x.cuda())
                        test_output = test_output.view(test_output.size(0), -1)
                    test_output = NNs[i].lrn(test_output)
                    test_output = test_output.cpu()
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    # print(test_y.data.numpy())
                    # print(pred_y)
                    accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum())\
                                    / float(test_y.size(0))
                    train_loss = NNs[i].loss.data.cpu().numpy()
                    
                    if is_dmd:
                        time.sleep(3)
                    
                    # record accuracy & time information
                    if accuracy > outputs[i].max_acc:
                        outputs[i].max_acc = accuracy
                        outputs[i].max_acc_loss = train_loss
                        outputs[i].max_acc_pred = pred_y
                        
                    if train_loss < outputs[i].min_loss:
                        outputs[i].min_loss = train_loss
                        outputs[i].min_loss_acc = accuracy
                        outputs[i].min_loss_pred = pred_y
                    print(outputs[i].dataset, '| Epoch:', epoch, '| train loss: %.4f' % train_loss,\
                          '| test accuracy: %.4f' % accuracy)
                    print('Curr Max Acc: %.4f' % outputs[i].max_acc, '- Max Acc loss: %.4f' % outputs[i].max_acc_loss)
                    print('Curr Min loss: %.4f' % outputs[i].min_loss, '- Min loss acc: %.4f' % outputs[i].min_loss_acc)
                    outputs[i].acc_evo[epoch] = accuracy
                    outputs[i].loss_evo[epoch] = train_loss
                if show_time:
                    print_time()
            data_cnt = data_cnt + batch_size
    
    if is_vsensor:
        global is_task_finish
        is_task_finish = 1
        time.sleep(1)
        vsensor_thread.join()
        outputs[0].vsensor_out = np.array(vsensor_out)
    
    if is_dmd:
        dmd_t.dmd_deinit()
    
if __name__ == "__main__":
    
    try:
        main()
    except BaseException as e:
        is_task_finish = 1
        time.sleep(1)
        if is_dmd:
            dmd_t.dmd_deinit()
        print(type(e))
        print(e)