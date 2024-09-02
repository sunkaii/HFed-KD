# HFed-KD

- OS Enviroment
```
Ubuntu 20.04
```
- Packages Requirement
```
torch==2.0.1
torchvision==0.15.2
torchaudio
torchinfo
scikit-learn
scipy
matplotlib
Flask==2.3.3
numpy
requests
tqdm
natsort
```
- Server config
```
ServerIP = "0.0.0.0"
ServerPort = 5001
# 選擇比例(論文實驗中都是1，全選)
SelectFrac = 1
MinNumberOfClient = 4
rounds = 999

# 使用同質or異質模型設定
model = "hetero_customCNN" #customCNN, hetero_customCNN
dataset = "cifar10" #fashionmnist,cifar10
local_ep = 1
local_bs = 32
spiltMethod = 1 # 0: default(total//MinNumberClient), 1: Non-iid
transform_type = 0 # 0: no transform, 1: with transform

# 伺服器和客戶端軟標籤互相擬合回合數
distillation_num_epochs = 4
# 每回合軟標籤訓練閾值
soft_labels_threshold = 4 # accept how many SL each Round
# 公共資料集數量
numbers_of_subset = 20000 # 50000
# 生成資料數量
generate_dataset_length = 0
# 軟標籤計算損失時CE和KL比例
balance_CEKL = 0.6 # balance parameter between CE and KL
temperature = 4 # soft labels temperature
# 模型選擇算法
modeltestMethod = "brute" # brute, greedy
# 伺服器每次訓練使用的軟標籤最小值
SL_train = 3

# 時間限制
time_restrict = 3600 # seconds
```
- Client config
```
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
```
- ACGAN的權重下載網址
https://drive.google.com/drive/folders/1wCOpv4KEcg0u19G25DXVqbILeVvl3616?usp=drive_link
- docker image網址
https://hub.docker.com/r/sunkaii/hfed-kd
- docker image執行指令
```
docker run -it --rm --gpus=all --network=host -v .:/app sunkaii/hfed-kd
```
- 伺服器端執行
```
cd Server/
python3 server.py
```
- 客戶端執行
```
cd Client/
python3 client.py
```
- 實驗結束後更新權重存放於./update資料夾內，實驗log存放於./history
- 準確率測試
```
cd Server/
python3 testmodel.py
# 有分每次更新(_.pt結尾)或是回合更新(round開頭)
````

## 注意事項
- Server部分紀錄client log部分有些地方是把client ip寫死
- Client部分資料集non-iid分布是根據參與數量寫的，目前只有寫3、4個客戶端
- select frac基本上是1