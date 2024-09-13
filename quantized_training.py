import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import Dataset, DataLoader
import gc  
import argparse 
import os
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import platform

# Ensure the quantization engine is set correctly
torch.backends.quantized.engine = 'fbgemm'

# Argument Parsing
parser = argparse.ArgumentParser(description='comsnets')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number to start training from')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number to end training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate'))
parser.add_argument('--gpu_list', type=str, default='0', help='GPU index')
parser.add_argument('--model_dir', type=str, default='model', help='directory to save trained or pre-trained models')
parser.add_argument('--data_dir', type=str, default='data', help='directory containing training data')
parser.add_argument('--log_dir', type=str, default='log', help='directory to save logs')
parser.add_argument('--save_interval', type=int, default=20, help='interval of saving model checkpoints')

args = parser.parse_args()

# Use the parsed arguments
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
gpu_list = args.gpu_list

# Set the device
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_list}")
    print("Using CUDA backend.")
else:
    device = torch.device("cpu")
    print("Using CPU backend.")

print(f"Running on: {device}")

# Define the ratio dictionary to map compression ratios to input sizes
ratio_dict = {
    1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327,
    35: 381, 40: 436, 45: 490, 50: 545, 60: 654, 70: 763, 80: 872
}

nrtrain = 88912 
batch_size = 32  

# Load training data
Training_data_Name = 'Training_Data.mat'
Training_data = sio.loadmat(os.path.join(args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']


class ResNet8(nn.Module):
    def __init__(self, num_layers):
        super(ResNet8, self).__init__()
        self.in_channels = 32
        self.quant = QuantStub()  
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, num_layers, stride=1)
        self.fc = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.dequant = DeQuantStub()  

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.fc(out)
        out = self.dequant(out)  
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

class CalibDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

    
calib_loader = DataLoader(
    dataset=CalibDataset(Training_labels, nrtrain),
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
)

def quantize_model(model,calib_loader):
    torch.cuda.empty_cache()
    model.eval()
    model.to('cpu')

    fusion_layers = [['conv1', 'bn1', 'relu']]
    for name, module in model.named_modules():
        if isinstance(module, BasicBlock):
            fusion_layers.append([f'{name}.conv1', f'{name}.bn1', f'{name}.relu1'])
            fusion_layers.append([f'{name}.conv2', f'{name}.bn2'])
    quant.fuse_modules(model, fusion_layers, inplace=True)

    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)

    with torch.no_grad():
        for i, data in enumerate(calib_loader):
            if i >= 1:
                break
            batch_x = data.view(-1, 1, 33, 33).to('cpu')
            model(batch_x)

    quant.convert(model, inplace=True)
    gc.collect()
    
    return model

def train_model(cs_ratio, model, log_file_name, model_dir):
    n_input = ratio_dict[cs_ratio]
    n_output = 1089
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if start_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(model_dir, f'net_params_{start_epoch}.pkl')))

    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        for data in calib_loader:
            batch_x = data.view(-1, 1, 33, 33).to(device)
            x_output = model(batch_x)

            loss_discrepancy = torch.mean((x_output - batch_x) ** 2)

            optimizer.zero_grad()
            loss_discrepancy.backward()
            optimizer.step()

        output_data = f"[{epoch_i:02d}/{end_epoch:02d}] CS Ratio: {cs_ratio}, Total Loss: {loss_discrepancy.item():.4f}\n"
        print(output_data)

        with open(log_file_name, 'a') as output_file:
            output_file.write(output_data)
        if epoch_i % args.save_interval == 0:
            save_path = os.path.join(model_dir, f"net_params_ratio_{cs_ratio}_epoch_{epoch_i}.pkl")
            torch.save(model.state_dict(), save_path)

    # Clear GPU memory after training
    torch.cuda.empty_cache()

# Main loop
if __name__ == "__main__":
    for cs_ratio in [20, 35, 45, 60, 70, 80]:
        model_dir = f"./{args.model_dir}/CS_ResNet8_ratio_{cs_ratio}"
        log_file_name = f"./{args.log_dir}/Log_CS_ResNet8_ratio_{cs_ratio}.txt"

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        print(f"Training for CS Ratio: {cs_ratio}")
        model = ResNet8(num_layers=2).to(device)
        train_model(cs_ratio, model, log_file_name, model_dir)

        # Move to CPU and quantize the model after training
        print("Quantizing the model...")
        model = quantize_model(model, calib_loader)
        torch.save(model.state_dict(), os.path.join(model_dir, "final_quantized_model.pkl"))
        print("Quantization complete.")

        # Clear any remaining resources
        del model
        gc.collect()
