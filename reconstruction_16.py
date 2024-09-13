import torch
import cv2
import numpy as np
import os
import glob
from skimage.metrics import structural_similarity as ssim
import gc
from time import time

# Configuration for directories and parameters
args = {
    'encoded_dir': '~/encoded',
    'result_dir': '~/result',
    'data_dir': '~/data',
    'test_name': 'Set11',
    'log_dir': '~/log'
}
cs_ratio = 20

# Prepare test directories
test_name = args['test_name']
test_dir = os.path.join(args['data_dir'], test_name)
filepaths = glob.glob(test_dir + '/*.png')

result_dir = os.path.join(args['result_dir'], test_name)
os.makedirs(result_dir, exist_ok=True)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    mse = np.mean((target_data - ref_data) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

with torch.no_grad():
    for img_no, imgName in enumerate(filepaths):
        start_time = time()
        try:
            Img = cv2.imread(imgName, cv2.IMREAD_COLOR)
            if Img is None or Img.size == 0:
                print(f"Error reading image {imgName}. Skipping.")
                continue

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()
            Iorg_y = Img_yuv[:, :, 0]
            encoded_filename = os.path.join(args['encoded_dir'], f"{os.path.basename(imgName)}.pt")
            x_output = torch.load(encoded_filename).half()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                X_rec = np.clip(x_output.cpu().numpy().squeeze()[:Iorg_y.shape[0], :Iorg_y.shape[1]], 0, 1).astype(np.float64)

            rec_PSNR = psnr(X_rec * 255, Iorg_y.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255, Iorg_y.astype(np.float64), data_range=255)

            print(f"[{img_no + 1}/{ImgNum}] PSNR: {rec_PSNR:.2f}, SSIM: {rec_SSIM:.4f}")

            Img_rec_yuv[:, :, 0] = X_rec * 255
            im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

            resultName = os.path.join(result_dir, os.path.basename(imgName))
            cv2.imwrite(f"{resultName}_ResNet8_ratio_{cs_ratio}.png", im_rec_rgb)

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM

        except Exception as e:
            print(f"Error processing {imgName}: {e}")

        finally:
            gc.collect()
            end_time = time() 
            elapsed_time = end_time - start_time
            print(f"Time taken for {os.path.basename(imgName)}: {elapsed_time:.2f} seconds")

# Output summary results
output_data = f"CS ratio is {cs_ratio}, Avg PSNR/SSIM for {test_name} is {np.mean(PSNR_All):.2f}/{np.mean(SSIM_All):.4f}"
print(output_data)

output_file_name = os.path.join(args['log_dir'], f"CS_ResNet8_ratio_{cs_ratio}.txt")
with open(output_file_name, 'a') as output_file:
    output_file.write(output_data)

print("Reconstruction completed.")

