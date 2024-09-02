ServerIP = "0.0.0.0" 
ServerPort = 5001
ClientIP = "0.0.0.0"
ClientPort = 5002
MinNumberOfClient = 3
rounds=10

model = "hetero_customCNN" #customCNN, hetero_customCNN
dataset = "cifar10" #fashionmnist,cifar10
local_ep=1
local_bs=8
spiltMethod = 0 # 0: default(total//MinNumberClient), 1: Non-iid
transform_type = 0 # 0: no transform, 1: use transform

distillation_num_epochs = 2
generate_dataset_length = 0
balance_CEKL = 0.5
temperature = 2