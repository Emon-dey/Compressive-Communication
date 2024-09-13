import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import glob
import gc
from time import time

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define the ResNet8 model
class ResNet8(nn.Module):
    def __init__(self, num_layers):
        super(ResNet8, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, num_layers, stride=1)
        self.fc = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.fc(out)
        return out

# Load the model
model = ResNet8(num_layers=2)
model.load_state_dict(torch.load('/home/emon/Desktop/OPINE-Net/model/q16.pkl', map_location='cpu'), strict=False)
model.eval().to('cuda').half()

# Configuration for directories
args = {
    'data_dir': '/home/emon/Desktop/OPINE-Net/data',
    'encoded_dir': '/home/emon/Desktop/OPINE-Net/encoded',
    'test_name': 'Set11',
}
os.makedirs(args['encoded_dir'], exist_ok=True)

# Prepare test directories
test_name = args['test_name']
test_dir = os.path.join(args['data_dir'], test_name)
filepaths = glob.glob(test_dir + '/*.png')

def imread_CS_py(img_y):
    row, col = img_y.shape
    row_new = int(np.ceil(row / 33) * 33)
    col_new = int(np.ceil(col / 33) * 33)
    img_pad = np.zeros((row_new, col_new), dtype=np.float32)
    img_pad[:row, :col] = img_y
    return img_y, row, col, img_pad, row_new, col_new

# List to store encoding times
encoding_times = []

# Encoding loop
with torch.no_grad():
    for img_no, imgName in enumerate(filepaths):
        start_time = time()
        try:
            Img = cv2.imread(imgName, cv2.IMREAD_COLOR)
            if Img is None or Img.size == 0:
                print(f"Error reading image {imgName}. Skipping.")
                continue

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Iorg_y = Img_yuv[:, :, 0]

            Iorg, row, col, Ipad, row_new, col_new = imread_CS_py(Iorg_y)
            Img_output = Ipad.reshape(1, 1, row_new, col_new) / 255.0

            batch_x = torch.from_numpy(Img_output).to('cuda').half()  
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                x_output = model(batch_x)

            encoded_filename = os.path.join(args['encoded_dir'], f"{os.path.basename(imgName)}.pt")
            torch.save(x_output, encoded_filename)

        except Exception as e:
            print(f"Error processing {imgName}: {e}")

        finally:
            del batch_x, x_output
            gc.collect()
            end_time = time()
            elapsed_time = end_time - start_time
            encoding_times.append(elapsed_time)
            print(f"Time taken for {os.path.basename(imgName)}: {elapsed_time:.2f} seconds")

average_time = sum(encoding_times) / len(encoding_times) if encoding_times else 0
print(f"Average encoding time per image: {average_time:.2f} seconds")

print("Encoding completed.")

