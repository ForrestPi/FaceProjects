import torch
import torchvision.transforms as T

class Config:
    # data preprocess
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    # training settings
    checkpoints = "checkpoints"
    restore_model = "last.pth"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = Config()