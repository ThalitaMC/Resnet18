# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
from torchvision import models

# Função para carregar a imagem e transformá-la para o formato aceito pela ResNet-18
def load_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Carregar o modelo ResNet-18 pré-treinado
modelow ='C:\\Users\\thali\\Resnet18\\Resnet18\\model_weights.pth'
#model=resnet18()
model= torch.load(modelow)
#model = 'model_weights.pth' #models.resnet18(pretrained=True)
model.eval()  # Colocar o modelo em modo de avaliação

# Carregar a imagem que você quer classificar
image_path = 'C:\\Users\\thali\\Resnet18\\val\\band\\BNE_486419.jpg'  # Substitua pelo caminho da sua imagem
image = load_image(image_path)

# Realizar a predição da classe da imagem
with torch.no_grad():
    outputs = model(image)

# Carregar os rótulos das classes (por exemplo, do ImageNet)
# Se necessário, substitua com os rótulos correspondentes ao seu conjunto de dados

with open('classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# Obter a classe com maior probabilidade
_, predicted = torch.max(outputs, 1)
predicted_label = labels[predicted.item()]

print(f"Class: {predicted_label}")