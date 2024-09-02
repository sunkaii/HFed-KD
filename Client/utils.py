import torch
from torchvision import datasets, transforms
import sys
import time
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torch.nn as nn
from models import *
from tqdm import tqdm

def get_datasets(args):
    if args.dataset == 'fashionmnist':
        data_dir = './data/fashionmnist/'
        if args.transform_type == 1:
            apply_transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            apply_transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        apply_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                        transform=apply_transform_train)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                        transform=apply_transform_test)
    elif args.dataset == "cifar10":
        data_dir = './data/cifar10/'
        if args.transform_type == 1:
            apply_transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            apply_transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        apply_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                         transform=apply_transform_test)
    return  train_dataset, test_dataset

def getTrainset_noTransform(args, generated_seed):
    torch.manual_seed(generated_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(generated_seed)
        torch.cuda.manual_seed_all(generated_seed)
    
    if args.dataset == "fashionmnist":
        data_dir = './data/fashionmnist/'
        if args.transform_type == 1:
            transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                        transform=transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                        transform=transform)    
    elif args.dataset == "cifar10":
        data_dir = './data/cifar10/'
        if args.transform_type == 1:
            transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomRotation(10),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        else:
            transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                            transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=transform)
    concat_dataset = ConcatDataset([train_dataset, test_dataset])
    return concat_dataset

def flask_start_message():
    for i in range(10):
        message = f'server start at {10-i}'
        sys.stdout.write(message)
        sys.stdout.flush()
        time.sleep(0.3)
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(0.3)
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(0.4)
        sys.stdout.write('\r' + ' ' * len(message) + ' ' * 4 + '\r')
        
class GeneratedImagesDataset(Dataset):
  def __init__(self, num_images, noise, label, generator, transform, seed=None):
    self.num_images = num_images
    self.noise = noise
    self.label = label
    self.transform = transform
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    generator.eval()
    with torch.no_grad():
      self.image = generator(self.noise, self.label)
    
  def __len__(self):
    return len(self.image)

  def __getitem__(self, idx):
    image = self.image[idx]
    if self.transform:
        image = self.transform(self.image[idx])
    
    label = self.label[idx].item()
    return image, label

def spilt_datasets(client_id, train_dataset, args):
    print(f'[i] Preparing datasets with method {args.spiltMethod}')
    if args.spiltMethod == 0:
        total_clients = args.MinNumberOfClient
        total_samples = len(train_dataset)
        samples_per_client = total_samples // total_clients
        start_idx = client_id * samples_per_client
        end_idx = (client_id + 1) * samples_per_client
        print(f'client {client_id}, start: {start_idx}, end: {end_idx}')
        return Subset(train_dataset, range(start_idx, end_idx))
    elif args.spiltMethod == 1: # Non-iid
        if args.MinNumberOfClient == 3:
            spilt_label = [[0, 1, 3, 5, 6, 7, 8, 9], [1, 2, 4, 5, 6, 7, 8, 9], [0, 2, 3, 4, 5, 6, 7, 8]]
        elif args.MinNumberOfClient == 4:
            spilt_label = [[1,3,4,7,9], [2,4,5,6,7], [2,3,5,7,8], [0,3,4,7,9]]
        spilt_indices = [i for i, (_, label) in enumerate(train_dataset) if label in spilt_label[client_id]]
        print(f'client {client_id}, using label {spilt_label[client_id]}')
        return Subset(train_dataset, spilt_indices)
    
def model_test_time(args):
    print("Model testing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "cifar10":
        data_dir = './data/cifar10/'
        if args.transform_type == 0:
            transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        elif args.transform_type == 1:
            transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=15),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif args.dataset == "fashionmnist":
        data_dir = './data/fashionmnist/'
        if args.transform_type == 0:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif args.transform_type == 1:
            transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        trainset = datasets.FashionMNIST(root=data_dir, train=True, download=True,
                                        transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss().to(device)
    
    warmup_model = CustomCNN_XS(args.dataset).to(device)
    optimizer = torch.optim.SGD(warmup_model.parameters(), lr=0.01, momentum=0.9)
    warmup_model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = warmup_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("WarnUp End")
    model_xs = CustomCNN_XS(args.dataset)
    model_s = CustomCNN_S(args.dataset)
    model_m = CustomCNN_M(args.dataset)
    model_l = CustomCNN_L(args.dataset)
    model_xl = CustomCNN_XL(args.dataset)
    model_list = [model_xs, model_s, model_m, model_l, model_xl]
    model_iter = []
    
    for model in model_list:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model = model.to(device)
        model.train()
        batch_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 5:
            for i, data in enumerate(tqdm(trainloader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_count += 1
                if time.time() - start_time >= 5:
                    break
        model_iter.append(batch_count)
    print(model_iter)
    return model_iter