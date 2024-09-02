import config
import threading
import requests
import time
import base64
from flask import Flask, request
from models import *
import pickle
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from utils import *
from tqdm import tqdm
import signal
import os
import socket
from timeutils import *
import random
import sys
from torchvision import transforms

server = None
uid = None
global_model = None
client_id = -1
pid = None
stopSignal = False
cnt_rounds = 1
aggFlag = False
train_dataset_noTransform = None
idle_time = timerUT()
getBytes = 0
round_seed = 0
subset_idx = []
subset_idx_train = []
generate_dataset_list = []
generate_seed = None
spilt_datasets_flag = False
client_dataset = None
new_soft_label_list = None

def generate_dataset():
    global generate_dataset_list, generate_seed
    if config.dataset == "cifar10":
        transform_gen = transforms.Compose([
            transforms.Resize((32, 32), antialias=True),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        weights = torch.load(f'./acgan_weights_cifar10/gen_900.pth')
    elif config.dataset == "fashionmnist":
        transform_gen = transforms.Compose([
            transforms.Resize((28, 28), antialias=True),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        weights = torch.load(f'./acgan_weights_fashionmnist/gen_900.pth')
    generator = Generator(dataset=config.dataset)
    generator.load_state_dict(weights)
    for i in range(10):
        print(f"Generating datasets of label {i}")
        if generate_seed is not None:
            torch.manual_seed(generate_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(generate_seed)
                torch.cuda.manual_seed_all(generate_seed)
        noise = torch.randn(config.generate_dataset_length, 100)
        labels = torch.ones(config.generate_dataset_length, dtype=torch.long) * i
        gen_dataset = GeneratedImagesDataset(config.generate_dataset_length, noise, labels, generator, transform=transform_gen, seed=generate_seed)
        generate_dataset_list.append(gen_dataset)
        
def train_soft_labels():
    global global_model, new_soft_label_list, train_dataset_noTransform, round_seed, aggFlag, subset_idx, generate_dataset_list
    print("Training soft labels")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset_global_dataset = Subset(train_dataset_noTransform, subset_idx)
    if config.generate_dataset_length > 0 :
        concat_dataset = ConcatDataset([subset_global_dataset, generate_dataset_list[0]])
        for i in range(1, len(generate_dataset_list)):
            concat_dataset = ConcatDataset([concat_dataset, generate_dataset_list[i]])
        subset_global_dataset = concat_dataset
    torch.manual_seed(round_seed+1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(round_seed+1)
        torch.cuda.manual_seed_all(round_seed+1)
    trainloader = DataLoader(subset_global_dataset, batch_size=config.local_bs, shuffle=True)
    copy_model = copy.deepcopy(global_model)
    copy_model = copy_model.to(device)
    copy_model.train()
    optimizer = torch.optim.SGD(copy_model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(config.distillation_num_epochs):
        epoch_seed = round_seed + epoch + 1
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader)):
            if aggFlag == True:
                print("[i] Stop getting soft labels")
                return
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = copy_model(inputs)
            loss_kl = F.kl_div(F.log_softmax(outputs / config.temperature, dim=1), new_soft_label_list[epoch][i*inputs.size(0): (i+1)*inputs.size(0)].to(device), reduction='batchmean') * (config.temperature ** 2)
            loss_ce = F.cross_entropy(outputs, labels)
            loss = (1 - config.balance_CEKL) * loss_ce + config.balance_CEKL * loss_kl
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print('Finish {} epoch, Loss: {:.6f}'.format(epoch + 1, running_loss / (len(train_dataset_noTransform))))
    global_model.load_state_dict(copy_model.state_dict())
    
def train():
    global global_model, uid, client_id, cnt_rounds, aggFlag
    global client_dataset, spilt_datasets_flag, new_soft_label_list, subset_idx_train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while aggFlag == False:
        if new_soft_label_list != None:
            train_soft_labels()
            new_soft_label_list = None
        if spilt_datasets_flag == False:
            train_dataset, test_dataset = get_datasets(config)
            train_dataset_concat = ConcatDataset([train_dataset, test_dataset])
            train_dataset_concat = Subset(train_dataset_concat, subset_idx_train)
            client_dataset = spilt_datasets(client_id, train_dataset_concat, config)
            spilt_datasets_flag = True
        trainloader = DataLoader(client_dataset, batch_size=config.local_bs, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        copy_model = copy.deepcopy(global_model)
        optimizer = torch.optim.SGD(copy_model.parameters(), lr=0.01, momentum=0.9)
        copy_model = copy_model.to(device)
        nsample = 0
        for epoch in range(config.local_ep):
            copy_model.train()
            running_loss = 0.0
            for i, data in enumerate(tqdm(trainloader), 0):
                if aggFlag == True:
                    print("[i] Stop train")
                    return
                nsample += 1
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = copy_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
        global_model.load_state_dict(copy_model.state_dict())
        torch.cuda.empty_cache()
        if aggFlag == False:
            print("[i] Getting Soft Labels")
            soft_labels = get_soft_labels()
            seed_ = random.randint(1, sys.maxsize)
            torch.manual_seed(seed_)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed_)
                torch.cuda.manual_seed_all(seed_) 
        if aggFlag == False:
            send_model(nsample, soft_labels)
            
def get_soft_labels():
    global global_model, uid, client_id, train_dataset_noTransform, round_seed, aggFlag, subset_idx, generate_dataset_list
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    subset_global_dataset = Subset(train_dataset_noTransform, subset_idx)
    if config.generate_dataset_length > 0:
            if config.spiltMethod == 1:
                    if config.MinNumberOfClient == 3:
                        spilt_label = [[0, 1, 3, 5, 6, 7, 8, 9], [1, 2, 4, 5, 6, 7, 8, 9], [0, 2, 3, 4, 5, 6, 7, 8]]
                    elif config.MinNumberOfClient == 4:
                        spilt_label = [[0, 1, 2, 3, 4, 5, 7, 9], [0, 1, 2, 3, 4, 6, 7, 9], [1, 2, 3, 4, 6, 7, 8, 9], [0, 2, 3, 4, 6, 7, 8, 9]]
                    print(f'client {client_id}, getting soft labels with label {spilt_label[client_id]}')
            concat_dataset = ConcatDataset([subset_global_dataset, generate_dataset_list[0]])
            for i in range(1, len(generate_dataset_list)):
                concat_dataset = ConcatDataset([concat_dataset, generate_dataset_list[i]])
            torch.manual_seed(round_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(round_seed)
                torch.cuda.manual_seed_all(round_seed)
            trainloader = DataLoader(concat_dataset, batch_size=config.local_bs, shuffle=True)
    else:
            torch.manual_seed(round_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(round_seed)
                torch.cuda.manual_seed_all(round_seed)
            if config.spiltMethod == 1:
                if config.MinNumberOfClient == 3:
                    spilt_label = [[0, 1, 3, 5, 6, 7, 8, 9], [1, 2, 4, 5, 6, 7, 8, 9], [0, 2, 3, 4, 5, 6, 7, 8]]
                elif config.MinNumberOfClient == 4:
                    spilt_label = [[0, 1, 2, 3, 4, 5, 7, 9], [0, 1, 2, 3, 4, 6, 7, 9], [1, 2, 3, 4, 6, 7, 8, 9], [0, 2, 3, 4, 6, 7, 8, 9]]
                print(f'client {client_id}, getting soft labels with label {spilt_label[client_id]}')
            trainloader = DataLoader(subset_global_dataset, batch_size=config.local_bs, shuffle=True)
    global_model.to(device)
    global_model.eval()
    soft_labels_epochs = []
    for epoch in range(config.local_ep):
        epoch_seed = round_seed + epoch
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        soft_labels = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(trainloader, 0)):
                if aggFlag == True:
                    print("[i] Stop train")
                    return
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = global_model(inputs)
                soft_labels.append(F.softmax(outputs / config.temperature, dim=1).cpu())
        soft_labels_epochs.append(torch.cat(soft_labels, dim=0))
        soft_labels_epochs.append(config.ClientIP)
    return soft_labels_epochs

def send_model(nsample, soft_labels):
    global uid
    rest = requests.post(f"http://{config.ServerIP}:{config.ServerPort}/model",
                                    data = {"model":base64.b64encode(pickle.dumps(soft_labels)),
                                            "nsample":nsample, "clientUID":uid})
        
class FLClient():
    def __init__(self):
        self.app = Flask(__name__)
        self.routes()
    def routes(self):
        @self.app.route('/identity', methods=['POST'])
        def identity():
            global uid, client_id
            uid = request.form['uid']
            client_id = int(request.form['client_id']) - 1
            print(uid)
            return 'get identity'
        @self.app.route('/configSync', methods=['POST'])
        def configSync():
            global idle_time, subset_idx, generate_seed, uid, subset_idx_train
            config.MinNumberOfClient = int(request.form['MinNumberOfClient'])
            config.model = request.form['model']
            config.dataset = request.form['dataset']
            config.local_ep = int(request.form['local_ep'])
            config.local_bs = int(request.form['local_bs'])
            config.rounds = int(request.form['rounds'])
            config.distillation_num_epochs = int(request.form['distillation_num_epochs'])
            subset_idx = pickle.loads(base64.b64decode(request.form['subset_idx']))
            config.generate_dataset_length = int(request.form['generate_dataset_length'])
            generate_seed = int(request.form['generate_seed'])
            config.spiltMethod = int(request.form['spiltMethod'])
            config.balance_CEKL = float(request.form['balance_CEKL'])
            config.temperature = int(request.form['temperature'])
            subset_idx_train = pickle.loads(base64.b64decode(request.form['subset_idx_train']))
            config.transform_type = int(request.form['transform_type'])
            init()
            if config.model == "hetero_customCNN":
                model_test = model_test_time(config)
                res = requests.post(f"http://{config.ServerIP}:{config.ServerPort}/modelTest",
                                        data = {"time":base64.b64encode(pickle.dumps(model_test)), "clientUID":uid})
            idle_time.segmentStart()
            return 'syncAck'
        @self.app.route('/model', methods=['POST'])
        def model():
            global cnt_rounds, aggFlag, idle_time, getBytes, round_seed, new_soft_label_list
            slList = request.form['softlist']
            cnt_rounds = int(request.form['cnt_rounds'])
            round_seed = int(request.form['round_seed'])
            getBytes += request.content_length
            idle_time.segmentEnd()
            new_soft_label_list = pickle.loads(base64.b64decode(slList))
            aggFlag = False
            time.sleep(5)
            train_thread = threading.Thread(target=train, daemon=True)
            print(f'[i] Round {cnt_rounds} start training')
            train_thread.start()
            return "receive"
        @self.app.route('/aggFlag', methods=['POST'])
        def aggFlag():
            global aggFlag, idle_time
            aggFlag = True
            idle_time.segmentStart()
            return 'ack'
        @self.app.route('/stop', methods=['POST'])
        def stop():
            global stopSignal
            print('Stopping')
            stopSignal = True
            return 'stop'
        @self.app.route('/model_selected', methods=['POST'])
        def model_selected():
            global global_model
            model_num = int(request.form['model_num'])
            if model_num == 0:
                global_model = CustomCNN_XS(dataset=config.dataset)
            elif model_num == 1:
                global_model = CustomCNN_S(dataset=config.dataset)
            elif model_num == 2:
                global_model = CustomCNN_M(dataset=config.dataset)
            elif model_num == 3:
                global_model = CustomCNN_L(dataset=config.dataset)
            elif model_num == 4:
                global_model = CustomCNN_XL(dataset=config.dataset)
            print(f"select model {model_num}")
            return "200"
    def run(self):
        config.ClientPort = find_available_port(config.ClientPort)
        self.app.run(debug=False, host=config.ClientIP, port=config.ClientPort)
        
def init():
    global global_model, train_dataset_noTransform
    if config.model == "customCNN":
        global_model = CustomCNN_L(dataset=config.dataset)
    train_dataset_noTransform = getTrainset_noTransform(config, generate_seed)
    if config.generate_dataset_length > 0:
        generate_dataset()
        
def ClientStopDet():
    global stopSignal, pid, idle_time, getBytes
    while True:
        if stopSignal == True:
            idle_time.segmentEnd()
            res = requests.post(f"http://{config.ServerIP}:{config.ServerPort}/idleTime",
                                    data = {"idle":base64.b64encode(pickle.dumps(idle_time)), "clientIP":config.ClientIP, "getBytes":getBytes})
            time.sleep(1)
            print('stop signal')
            os.kill(pid, signal.SIGINT)
        time.sleep(5)
        
def FLClientHost():
    serverT = FLClient()
    serverT.run()

def find_available_port(starting_port):
    port = starting_port
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((config.ClientIP, port))
            sock.close()
            break
        except OSError:
            port += 1
            if port > 65535:
                raise ValueError("No available ports found.")
    return port

if __name__ == "__main__":
    pid = os.getpid()
    clientStopDet = threading.Thread(target=ClientStopDet, daemon=True)
    clientFlaskServer = threading.Thread(target=FLClientHost, daemon=True)
    clientFlaskServer.start()
    clientStopDet.start()
    flask_start_message()
    while True:
        if uid == None:
            res = requests.post(f"http://{config.ServerIP}:{config.ServerPort}/join", 
                                    data={"client_addr":config.ClientIP, "client_port":config.ClientPort})
        else:
            break
        time.sleep(10)
        if uid == None:
            print('resend')
    clientStopDet.join()
    clientFlaskServer.join()