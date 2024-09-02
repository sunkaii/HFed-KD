ServerIP = "0.0.0.0" 
ServerPort = 5001
SelectFrac = 1
MinNumberOfClient = 4
rounds = 999

model = "hetero_customCNN" #customCNN, hetero_customCNN
dataset = "cifar10" #fashionmnist,cifar10
local_ep = 1
local_bs = 32
spiltMethod = 1 # 0: default(total//MinNumberClient), 1: Non-iid
transform_type = 0 # 0: no transform, 1: with transform

distillation_num_epochs = 4
soft_labels_threshold = 4 # accept how many SL each Round
numbers_of_subset = 20000 # 50000
generate_dataset_length = 0
balance_CEKL = 0.6 # balance parameter between CE and KL
temperature = 4 # soft labels temperature
modeltestMethod = "brute" # brute, greedy
SL_train = 3

time_restrict = 3600 # seconds