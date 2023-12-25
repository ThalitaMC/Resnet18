# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from PIL import Image

# Define o dispositivo para executar o modelo (CPU ou GPU, se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega o modelo pré-treinado
modelo = torch.load('model_weights.pth', map_location=device)
modelo.eval()  # Define o modelo para o modo de avaliação

# Define transformações para redimensionar e normalizar as imagens de entrada
transformacao = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Função para classificar uma imagem
def classificar_imagem(image_path):
    # Carrega a imagem
    imagem = Image.open(image_path)
    imagem = transformacao(imagem).unsqueeze(0)  # Aplica transformações na imagem

    # Move a imagem para o dispositivo
    imagem = imagem.to(device)

    # Passa a imagem pelo modelo
    with torch.no_grad():
        saida = modelo(imagem)

    # Obtém as probabilidades previstas e a classe com maior probabilidade
    probabilidades = torch.nn.functional.softmax(saida[0], dim=0)
    classe_prevista = torch.argmax(probabilidades).item()

    return classe_prevista, probabilidades[classe_prevista].item(), probabilidades

# Caminho para a imagem que deseja classificar
caminho_imagem = 'C:\\Users\\thali\\Resnet18\\val\\band\\BNE_486419.jpg'

# Classifica a imagem
classe, probabilidade, probabilidades = classificar_imagem(caminho_imagem)

# Imprime o resultado
print(f"Classe prevista: {classe}")
print(f"Probabilidade: {probabilidade}")
print(f"Todas as probabilidades: {probabilidades}")
