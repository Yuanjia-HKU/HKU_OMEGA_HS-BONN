import torch
import cv2 as cv
import numpy as np
import torch.nn as nn
from scipy import stats
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

# Use GPU
torch.cuda.set_device(0)

dir_path = './Engine_scheduler/'
weight_path = dir_path + 'engine_net.pth'
sample_data_num = 10

width_ds = 26
length_ds = 32
X_t = np.load(dir_path + 'Dataset_bin_t.npy')
y = np.load(dir_path + 'Dataset_label.npy')
class Engine_train_data(Data.Dataset):
    def __init__(self, dataset_t, label_t):
        self.data = torch.FloatTensor(dataset_t)
        self.label = torch.LongTensor(label_t)

    def __getitem__(self,index):
        return (self.data[index]).cuda(),(self.label[index]).cuda()
    
    def __len__(self):
        return len(self.data)

class Engine_NN(nn.Module):
    def __init__(self):
        super(Engine_NN, self).__init__()
        self.fc1 = nn.Linear(width_ds * length_ds, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 4)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def dataset_preprocess(dataset):
    out = np.zeros((len(dataset), width_ds, length_ds))
    for i in range(len(dataset)):
        out[i] = cv.resize(dataset[i], (length_ds, width_ds))
    out = out.reshape((-1, width_ds * length_ds))
    return out

def train_engine_scheduler():
    print("Start engine scheduler training!")
    # Downsample the dataset
    X = dataset_preprocess(X_t)
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    train_loader = Data.DataLoader(dataset = Engine_train_data(X_train, y_train), batch_size=50, shuffle = True)
    
    model = Engine_NN().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 50
    acc_max = 0
    
    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            model.train()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.cuda())
            val_outputs = val_outputs.cpu()
            val_loss = criterion(val_outputs, y_val_tensor)
            pred_y = torch.max(val_outputs, 1)[1].data.numpy()
            val_acc = ((pred_y == y_val_tensor.data.numpy()).astype(int).sum())\
                            / float(y_val_tensor.size(0))
        
        print('Train Epoch:', epoch, '| train loss: %.4f' % val_loss,\
              '| test accuracy: %.4f' % val_acc)
        
        if val_acc > acc_max:
            acc_max = val_acc
            torch.save(model.state_dict(), weight_path)
    print("Trained weights saved to path ", weight_path, "with max acc %.4f" % acc_max)

def Engine_schedule(data_in):
    data_sample = dataset_preprocess(data_in)
    
    model = Engine_NN().cuda()
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    
    sample_pred = np.zeros((sample_data_num))
    for idx in range(0, sample_data_num):
        sample_in = data_sample[idx]
        sample_in_tensor = torch.FloatTensor(sample_in).cuda()
        
        with torch.no_grad():
            output = model(sample_in_tensor)
            sample_pred[idx] = torch.argmax(output, dim = 0)
    
    dataset_id = stats.mode(sample_pred).mode
    return dataset_id.astype('int')

if __name__ == "__main__":
    try:
        train_engine_scheduler()
    except BaseException as e:
        print(type(e))
        print(e)