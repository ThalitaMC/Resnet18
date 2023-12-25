import torchvision.models as models
import torch.nn as nn

def build_model(pretrained=False, fine_tune=True, num_classes=4):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT)
        folder = 'C:\\Users\\thali\\Resnet18\\outputs'
        nn.save(model.state_dict(), folder)
        
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights')
        model = models.resnet18(weights=None)
        folder = 'C:\\Users\\thali\\Resnet18\\outputs'
        nn.save(model.state_dict(), folder)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
            
    # Change the final classification head, it is trainable.
    model.fc = nn.Linear(512, num_classes)
    folder = 'C:\\Users\\thali\\Resnet18\\outputs'
    nn.save(model.state_dict(), folder)
    return model

if __name__ == '__main__':
    model = build_model(num_classes=4)
    folder = 'C:\\Users\\thali\\Resnet18\\outputs'
    nn.save(model.state_dict(), folder)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
