import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
#from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize

plt.style.use('ggplot')

def get_data(batch_size=16):
    # CIFAR10 training dataset.
    #TrainFolder = "C:\\Users\\thali\\Resnet50\\train"
    #ValidateFolder= "C:\\Users\\thali\\Resnet50\\val"

    data_dirt = 'C:\\Users\\thali\\Resnet18\\train'
    data_dirv = 'C:\\Users\\thali\\Resnet18\\val'
    transform = Compose([
    Resize((369, 369)),  # Redimensiona as imagens para o tamanho desejado
    ToTensor()  # Converte as imagens para tensores
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza os tensores
    ])
    datasett = ImageFolder(root=data_dirt, transform=transform)
    batch_size = 16
    dataset_train = DataLoader(datasett, batch_size=batch_size, shuffle=True)

    datasetv = ImageFolder(root=data_dirv, transform=transform)
    batch_size = 16
    dataset_valid = DataLoader(datasetv, batch_size=batch_size, shuffle=False)
    return dataset_train, dataset_valid

def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_loss.png'))
