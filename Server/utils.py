import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset
from models import *
from itertools import product

def aggregate_SL(sl):
    average_SL = []
    for tensors in zip(*sl):
        average_tensor = sum(tensors) / len(tensors)
        average_SL.append(average_tensor)
    return average_SL

def client_selection(clients, n):
    return random.sample(list(clients), int(n*len(clients)))

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

def findOptimalModelSet(connected_user, args):
    if args.modeltestMethod == "brute":
        model_times_per_client = [client.model_time for client in connected_user.values()]
        all_possible_assignments = list(product(*[range(len(times)) for times in model_times_per_client]))
        min_distance = float('inf')
        optimal_assignment_indices = None
        
        for assignment in all_possible_assignments:
            current_assignment_batches = [connected_user[uid].model_time[index] for uid, index in zip(connected_user.keys(), assignment)]
            avg_batch = sum(current_assignment_batches) / len(current_assignment_batches)
            distance = sum(abs(batch - avg_batch) for batch in current_assignment_batches)
            if distance < min_distance:
                min_distance = distance
                optimal_assignment_indices = assignment
        
        for uid, index in zip(connected_user.keys(), optimal_assignment_indices):
            connected_user[uid].model = index
        
        return min_distance, {client.uid: client.model for client in connected_user.values()}
    elif args.modeltestMethod == "greedy":
        model_times_per_client = [client.model_time for client in connected_user.values()]
        global_average = sum(sum(model_time_list) for model_time_list in model_times_per_client) / sum(len(model_time_list) for model_time_list in model_times_per_client)
        
        min_distance_list = []
        optimal_assignment_indices_list = []
        
        for client in connected_user.values():
            min_distance = float('inf')
            min_assignment = None
            for idx, time in enumerate(client.model_time):
                if(abs(time - global_average) < min_distance):
                    min_distance = abs(time - global_average)
                    min_assignment = idx
            min_distance_list.append(min_distance)
            optimal_assignment_indices_list.append(min_assignment)
        
        optimal_assignment_indices = tuple(optimal_assignment_indices_list)
        for uid, index in zip(connected_user.keys(), optimal_assignment_indices):
            connected_user[uid].model = index
            
        return min_distance_list, {client.uid: client.model for client in connected_user.values()}