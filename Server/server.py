import threading
import requests
import config
import uuid
import pickle
from flask import Flask, request
import base64
import time
import copy
from utils import *
from models import *
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn.functional as F
import signal
import os
import datetime
from timeutils import *
import sys
import random
from torchvision import transforms
from tqdm import tqdm

connected_user = {}
connected_user_lock = threading.Lock()
FL_State = "waiting"
cnt_rounds = 1
selected_clients = []
global_model = None
train_dataset = None
train_dataset_noTransform = None
soft_labels_list_lock = threading.Lock()
new_soft_labels_list_lock  = threading.Lock()
soft_labels_list = []
train_soft_labels_threads = None
soft_labels_train_count = 0
acceptSL_flag = True         # 是否接收soft labels
total_time = timerUT()
round_time = timerUT()
idle_time = timerUT()
idle_Client = []
cnt_update = 0
getBytes = 0                 # server接收到的總數據量
getBytes_list = []           # server每次收到SL的累加數據量
time_restrict_flag = False
round_seed = 0
subset_idx = []
subset_idx_train = []
generate_dataset_list = []
generate_seed = random.randint(1, sys.maxsize)
new_soft_label_list = None
client_update_count = [0, 0, 0, 0] #[181, 182, 183, 187]
getBytes_client = [0, 0, 0, 0] #[181, 182, 183, 187]

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
    global soft_labels_list, soft_labels_list_lock, train_dataset_noTransform, global_model, soft_labels_train_count, acceptSL_flag, total_time, cnt_update, idle_time, round_seed
    global generate_dataset_list, client_update_count, idle_time, getBytes, getBytes_list
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    while len(soft_labels_list) >= config.SL_train and acceptSL_flag == True:
        if idle_time.segment[-1]["end"] == None:
            idle_time.segmentEnd()
        print(f"{soft_labels_train_count} SLs have been train this round.")
        select_SL = []
        select_SL_idx = []
        soft_labels_list_lock.acquire()
        getBytes_list.append(getBytes)
        for i in range(len(soft_labels_list)):
            select_IP = soft_labels_list[i][-1]
            if select_IP == "0.0.0.0":
                client_update_count[0] += 1
            elif select_IP == "0.0.0.0":
                client_update_count[1] += 1
            elif select_IP == "0.0.0.0":
                client_update_count[2] += 1
            elif select_IP == "0.0.0.0": 
                client_update_count[3] += 1
            soft_labels_list[i].pop()
            select_SL.append(copy.deepcopy(soft_labels_list[i]))
            select_SL_idx.append(i)
        average_SL = aggregate_SL(select_SL)
        select_SL = []
        select_SL.append([average_SL, len(average_SL) - 1])
        soft_labels_list_lock.release()
        subset_global_dataset = Subset(train_dataset_noTransform, subset_idx)
        if config.generate_dataset_length > 0:
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
            trainloader = DataLoader(subset_global_dataset, batch_size=config.local_bs, shuffle=True)
        global_model = global_model.to(device)
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(config.local_ep):
            running_loss = 0.0
            epoch_seed = round_seed + epoch
            torch.manual_seed(epoch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(epoch_seed)
                torch.cuda.manual_seed_all(epoch_seed)
            for i, data in tqdm(enumerate(trainloader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = global_model(inputs)
                loss_kl = F.kl_div(F.log_softmax(outputs / config.temperature, dim=1), select_SL[0][0][epoch][i*inputs.size(0): (i+1)*inputs.size(0)].to(device), reduction='batchmean') * (config.temperature ** 2)
                for idx in range(1, len(select_SL)):
                    loss_kl += F.kl_div(F.log_softmax(outputs / config.temperature, dim=1), select_SL[idx][0][epoch][i*inputs.size(0): (i+1)*inputs.size(0)].to(device), reduction='batchmean') * (config.temperature ** 2)
                loss_ce = F.cross_entropy(outputs, labels)
                loss = (1 - config.balance_CEKL) * loss_ce + config.balance_CEKL * loss_kl
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        print('Finish {} epoch, Loss: {:.6f}'.format(epoch + 1, running_loss / (len(train_dataset_noTransform))))
        cnt_update += 1
        total_time.addStamp(info=f"update {cnt_update}")
        soft_labels_list_lock.acquire()
        for idx in range(len(select_SL_idx)):
            soft_labels_list.pop(0)
        soft_labels_list_lock.release()
        soft_labels_train_count += 1
        if not os.path.exists('./update'):
            os.makedirs('./update')
        torch.save(global_model, f'./update/{cnt_update}_{len(select_SL)}_{select_IP}_.pt')
        if soft_labels_train_count >= config.MinNumberOfClient * config.soft_labels_threshold:
            acceptSL_flag = False
    if acceptSL_flag == True and idle_time.segment[-1]["end"] != None:
        idle_time.segmentStart()
        
def get_soft_labels():
    global global_model, train_dataset_noTransform, round_seed, subset_idx, generate_dataset_list
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    subset_global_dataset = Subset(train_dataset_noTransform, subset_idx)
    if config.generate_dataset_length > 0:
        concat_dataset = ConcatDataset([subset_global_dataset, generate_dataset_list[0]]) # 先取label = 0
        for i in range(1, len(generate_dataset_list)):
            concat_dataset = ConcatDataset([concat_dataset, generate_dataset_list[i]]) # 再取label 1~9
        subset_global_dataset = concat_dataset
    torch.manual_seed(round_seed+1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(round_seed+1)
        torch.cuda.manual_seed_all(round_seed+1)
    trainloader = DataLoader(subset_global_dataset, batch_size=config.local_bs, shuffle=True)
    global_model = global_model.to(device)
    global_model.eval()
    soft_label_epochs = []
    for epoch in range(config.distillation_num_epochs):
        epoch_seed = round_seed + epoch + 1
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        soft_labels = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(trainloader, 0)):
                inputs, labels = data
                inputs, lables = inputs.to(device), labels.to(device)
                outputs = global_model(inputs)
                soft_labels.append(F.softmax(outputs / config.temperature, dim=1).cpu())
        soft_label_epochs.append(torch.cat(soft_labels, dim=0))
    return soft_label_epochs

class FLClientInfo:
    def __init__(self, _addr, _port, _uid):
        self.uid = _uid
        self.addr = _addr
        self.port = _port
        self.stat = 'connected'
        self.model_time = None
        self.model = None
        
class FLserver:
    def __init__(self):
        self.app = Flask(__name__)
        self.routes()
    def routes(self):
        @self.app.route('/join', methods=['POST'])
        def join():
            global connected_user, connected_user_lock
            uid = str(uuid.uuid4())
            client_addr = request.form['client_addr']
            client_port = request.form['client_port']
            print(client_addr, client_port)
            client = FLClientInfo(client_addr, client_port, uid)
            connected_user_lock.acquire()
            connected_user[uid] = client
            print(f"{client.uid} connected")
            res = requests.post(f"http://{client.addr}:{client.port}/identity",
                               data={"uid":uid, "client_id":len(connected_user)})
            connected_user_lock.release()
            return 'welcome'
        @self.app.route('/model', methods=['POST'])
        def model():
            global connected_user, connected_user_lock, train_soft_labels_threads
            global soft_labels_list, soft_labels_list_lock, acceptSL_flag, idle_time, getBytes
            client_weight_p = request.form['model']
            nsample = request.form['nsample']
            clientUID = request.form['clientUID']
            print(f"Received Model from {clientUID} with nsampled {nsample}")
            soft_labels = pickle.loads(base64.b64decode(client_weight_p))
            soft_labels_list_lock.acquire()
            getBytes += request.content_length
            soft_labels_list.append(soft_labels)
            soft_labels_list_lock.release()
            if not train_soft_labels_threads.is_alive():
                train_soft_labels_threads = threading.Thread(target=train_soft_labels, daemon=True)
                print(f"[i] Start SL training...")
                train_soft_labels_threads.start()
            connected_user_lock.acquire()
            connected_user[clientUID].stat = "ready"
            connected_user_lock.release()
            return 'receive'
        @self.app.route('/idleTime', methods=['POST'])
        def idleTime():
            global idle_Client, soft_labels_list_lock, getBytes_client
            idle_ = request.form['idle']
            idle = pickle.loads(base64.b64decode(idle_))
            client_bytes = int(request.form['getBytes'])
            clientIP = request.form['clientIP']
            soft_labels_list_lock.acquire()
            idle_Client.append({"clientIP":clientIP, "idle_time":idle})
            if clientIP == "0.0.0.0":
                getBytes_client[0] = client_bytes
            elif clientIP == "0.0.0.0":
                getBytes_client[1] = client_bytes
            elif clientIP == "0.0.0.0":
                getBytes_client[2] = client_bytes
            elif clientIP == "0.0.0.0":
                getBytes_client[3] = client_bytes
            soft_labels_list_lock.release()
            return "200"
        @self.app.route('/modelTest', methods=['POST'])
        def modelTest():
            global connected_user
            clientUID = request.form['clientUID']
            model_time = pickle.loads(base64.b64decode(request.form['time']))
            connected_user[clientUID].model_time = model_time
            return "200"
            
    def run(self):
        self.app.run(debug=False, host=config.ServerIP, port=config.ServerPort)
        
def syncClientConfig(uid):
    global connected_user, subset_idx, generate_seed, subset_idx_train
    while True:
        res = requests.post(f"http://{connected_user[uid].addr}:{connected_user[uid].port}/configSync",
                    data = {"model":f'{config.model}', "dataset":f'{config.dataset}', 
                            "local_ep":f'{config.local_ep}', "local_bs":f'{config.local_bs}', "MinNumberOfClient":f'{config.MinNumberOfClient}',
                            "rounds":f'{config.rounds}', "distillation_num_epochs":f'{config.distillation_num_epochs}',
                            "subset_idx":base64.b64encode(pickle.dumps(subset_idx)), "generate_dataset_length":f'{config.generate_dataset_length}', 
                            "generate_seed":f'{generate_seed}', "spiltMethod":f'{config.spiltMethod}', "transform_type":f'{config.transform_type}',
                            "temperature":f'{config.temperature}', "balance_CEKL":f'{config.balance_CEKL}', "subset_idx_train":base64.b64encode(pickle.dumps(subset_idx_train))})
        if res.status_code != 200:
            print('syncConfig fail')
        else:
            break
        time.sleep(3)
        
def init():
    global global_model, train_dataset, test_dataset, train_soft_labels_threads, train_dataset_noTransform, subset_idx, generate_seed, subset_idx_train
    if config.model == "customCNN" or config.model == "hetero_customCNN":
        global_model = CustomCNN_L(dataset=config.dataset)
    train_dataset, test_dataset = get_datasets(config)
    train_dataset_noTransform = getTrainset_noTransform(config, generate_seed)
    train_soft_labels_threads = threading.Thread(target=train_soft_labels, daemon=True)
    subset_idx = torch.randperm(len(train_dataset_noTransform)).tolist()
    subset_idx_train = subset_idx[config.numbers_of_subset:]
    subset_idx = subset_idx[:config.numbers_of_subset]
    if config.generate_dataset_length > 0:
        generate_dataset()
        
def FLScheduler():
    global FL_State, connected_user, cnt_rounds, client_update_count
    global selected_clients, global_model, getBytes_list
    global  total_time, round_time, idle_time, idle_Client, getBytes, round_seed, new_soft_labels_list_lock, new_soft_label_list
    global  train_dataset, test_dataset, soft_labels_list, soft_labels_list_lock, soft_labels_train_count, acceptSL_flag, time_restrict_flag, cnt_update, getBytes_client
    while True:
        if cnt_rounds > config.rounds:
            total_time.end()
            for uid in connected_user:
                res = requests.post(f"http://{connected_user[uid].addr}:{connected_user[uid].port}/stop")
            print("Wait 10 seconds to write log..")
            time.sleep(10)
            print(f"Total time: {total_time.duration()} seconds.")
            roundTime_list, idleTime_list = [], []
            for i in range(len(round_time.segment)):
                if round_time.segment[i]["end"] != None:
                    roundTime_list.append(round_time.segmentDuration(idx=i))
            for i in range(len(idle_time.segment)):
                if idle_time.segment[i]["end"] != None:
                    idleTime_list.append(idle_time.segmentDuration(idx=i))
            update_list = []
            for i in range(len(total_time.stamp)):
                update_list.append(round(total_time.stamp[i]["time"] - total_time.start_time, 2))
            client_idle_total = [0, 0, 0, 0]
            for client in range(len(connected_user)):
                sum_idle = 0
                for i in range(len(idle_Client[client]["idle_time"].segment)):
                    sum_idle += idle_Client[client]["idle_time"].segmentDuration(idx=i)
                if idle_Client[client]["clientIP"] == "0.0.0.0":
                    client_idle_total[0] = round(sum_idle, 2)
                elif idle_Client[client]["clientIP"] == "0.0.0.0":
                    client_idle_total[1] = round(sum_idle, 2)
                elif idle_Client[client]["clientIP"] == "0.0.0.0":
                    client_idle_total[2] = round(sum_idle, 2)
                elif idle_Client[client]["clientIP"] == "0.0.0.0":    
                    client_idle_total[3] = round(sum_idle, 2)
            log_ = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
            if not os.path.exists('./history'):
                os.makedirs('./history')
            with open(f'./history/dis_{config.distillation_num_epochs}_{config.soft_labels_threshold}_{config.numbers_of_subset}_{config.generate_dataset_length}_{config.balance_CEKL}_{config.temperature}_{config.SL_train}_{config.modeltestMethod}_{config.dataset}_{log_}.log', 'w') as f:
                f.write('roundTime = ' + str(roundTime_list) + '\n')
                f.write('update_since_start = ' + str(update_list) + '\n')
                f.write('idleTime_server = ' + str(idleTime_list) + '\n')
                f.write('idleTime_client = ' + str(client_idle_total) + '\n')
                f.write('BytesServer = ' + str(getBytes) + '\n')
                f.write('BytesList_Server = ' + str(getBytes_list) + '\n')
                f.write('BytesClients =' + str(getBytes_client) + '\n')
                f.write('Totaltime = ' + str(total_time.duration()) + '\n')
                f.write('Client_selection_count = ' + str(client_update_count) + '\n')
            break
        if FL_State == "waiting" and len(connected_user) >= config.MinNumberOfClient:
            if cnt_rounds == 1:
                total_time.start()
                syncClientThread = []
                for uid in connected_user.keys():
                    try:
                        thr = threading.Thread(target=syncClientConfig, args=(uid,))
                        syncClientThread.append(thr)
                        thr.start()
                    except Exception as e:
                        print(e)
                        print(f'[e] Sync client error')
                        while True:
                            thr = syncClientThread[-1]
                            thr.raise_exception()
                            syncClientThread = syncClientThread[:-2]
                            print(f'[i] Model Test to {uid}')
                            thr = threading.Thread(target=syncClientConfig, args=(uid,))
                            syncClientThread.append(thr)
                            thr.start()
                            time.sleep(20)
                for thr in syncClientThread:
                    print("[i] Waiting for Model Test join")
                    thr.join()
                if config.model == "hetero_customCNN":
                    min_distance, optimal_model_indices = findOptimalModelSet(connected_user, config)
                    print("最小距離總和:", min_distance)
                    print(f"optimal_model_indices: {optimal_model_indices}")
                    for uid, index in optimal_model_indices.items():
                        connected_user[uid].model = index
                    for uid in connected_user.keys():
                        print(f"client {connected_user[uid].addr} use model {connected_user[uid].model}")
                        res = requests.post(f"http://{connected_user[uid].addr}:{connected_user[uid].port}/model_selected",
                                data = {"model_num":connected_user[uid].model})
            selected_clients = client_selection(connected_user, config.SelectFrac)
            print(f"Client {selected_clients} has been chose")
            FL_State = "preparing"
        if FL_State == "preparing" and len(connected_user) >= config.MinNumberOfClient:
            round_seed = random.randint(1, sys.maxsize)
            acceptSL_flag = True
            print(f"[i] Round {cnt_rounds}, Server will train in 3 sec")
            time.sleep(3)
            round_time.segmentStart(info=cnt_rounds)
            sl = get_soft_labels()
            new_soft_labels_list_lock.acquire()
            new_soft_label_list = sl
            new_soft_labels_list_lock.release()
            idle_time.segmentStart()
            FL_State = "training"
            sendModelThreads = []
            for uid in selected_clients:
                try:
                    thr = threading.Thread(target=sendModelHandler, args=(uid, round_seed))
                    sendModelThreads.append(thr)
                    thr.start()
                except Exception as e:
                    print(e)
                    print(f'[e] Sync client error')
                    while True:
                        thr = sendModelThreads[-1]
                        thr.raise_exception()
                        sendModelThreads = sendModelThreads[:-2]
                        print(f'[i] Send model to {uid}')
                        thr = threading.Thread(target=sendModelHandler, args=(uid, round_seed))
                        sendModelThreads.append(thr)
                        thr.start()
                        time.sleep(20)
            # Waiting sendModel Thread join
            for thr in sendModelThreads:
                print("[i] Waiting for join")
                thr.join()
        if FL_State == "training" and soft_labels_train_count >= config.MinNumberOfClient * config.soft_labels_threshold:
            acceptSL_flag = False
            for uid in connected_user:
                res = requests.post(f"http://{connected_user[uid].addr}:{connected_user[uid].port}/aggFlag")
            train_soft_labels_threads.join()
            round_time.segmentEnd()
            torch.save(global_model, f'./update/round_{cnt_rounds}.pt')
            soft_labels_list_lock.acquire()
            soft_labels_list.clear()
            soft_labels_list_lock.release()
            soft_labels_train_count = 0
            cnt_rounds += 1
            FL_State = "waiting"
            if idle_time.segment[-1]["end"] == None:
                idle_time.segmentEnd()
                print(f'=== idle time THR === {idle_time.segmentDuration(idx=-1)}')
        if config.time_restrict != 0 and (time.time() - total_time.start_time) > config.time_restrict and time_restrict_flag == False:
            print(f"\n [i] Time exceed {config.time_restrict}, last round \n ")
            soft_labels_train_count = config.MinNumberOfClient * config.soft_labels_threshold
            cnt_rounds = config.rounds
            time_restrict_flag = True
        time.sleep(1)
    os.kill(os.getpid(), signal.SIGINT)
    
def sendModelHandler(uid, round_seed):
    global connected_user, new_soft_label_list, cnt_rounds
    print(f"send global soft label to {uid}")
    res = requests.post(f"http://{connected_user[uid].addr}:{connected_user[uid].port}/model",
                    data = {"softlist":base64.b64encode(pickle.dumps(new_soft_label_list)), "cnt_rounds":cnt_rounds, "round_seed":round_seed})
    
if __name__ == "__main__":
    init()
    fl_scheduler = threading.Thread(target=FLScheduler, daemon=True)
    fl_scheduler.start()
    server = FLserver()
    server.run()
    fl_scheduler.join()