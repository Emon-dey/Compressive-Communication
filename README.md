# Compressive-Communication
This repository includes the codes for the paper paper under review named 'Scalable and Efficient Adaptive Data Transmission for Resource-Constrained Robotic Networks'. In this work we propose an adaptive transmission framework to vary the sliding window size and also the data quantizationn bit size based on the physical layer properties. 

# Running transmission and receiver codes
Prerequisities:
```
1. pip install scikit-image
2. cuda support is required to execute some of the modules
2. Pytorch

```
Clone the github repository to the host computed and modify the directories for data, model, log and result accordingly. The code can be executed in a single device or if have have multiple devices, you can setup one as transmistter and other one as receiver.
```
1. Open the 'comsnets_run.ipynb' file
2. In the cell, run either encoding.py or encoding_16.py files for int8 or FP16 quantized encided feature geeneration.
3. For receiver, run either reconstruction.py or reconstruction_16.py for the respective quantization bit size.

```
# Training the model

```
Run eithre quantized_training.py or quantized_training_16.py to generate the lightweight models for int8 and FP16 quantization.
```
