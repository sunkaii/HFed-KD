import os
import torch
from models import *
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from natsort import natsorted
import config

FOLDER_PATH = "./update"

test_accuracy = []
test_loss = []

if config.dataset == "cifar10":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
elif config.dataset == "fashionmnist":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5)),
    ])
    testset = torchvision.datasets.FashionMNIST(root='./data/fashionmnist', train=False, download=True, transform=transform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_list = natsorted(os.listdir(FOLDER_PATH))

for filename in file_list:
    file_path = os.path.join(FOLDER_PATH, filename)
    if filename.endswith('_.pt'):
    #if filename.startswith('round'):
        print(f"[i] Testing {filename}...")
        model = torch.load(file_path)
        model = model.to(device)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        average_loss = total_loss / len(testloader)
        accuracy = 100 * correct / total

        print(f"Accuracy on the test set: {accuracy}%")
        print(f"Average loss on the test set: {average_loss}")
        test_accuracy.append(accuracy)
        test_loss.append(round(average_loss, 3))

with open(f"{FOLDER_PATH}/result.log", "w") as f:
    f.write('accuracy = ' + str(test_accuracy) + '\n')
    f.write('loss = ' + str(test_loss) + '\n')
    f.close()
