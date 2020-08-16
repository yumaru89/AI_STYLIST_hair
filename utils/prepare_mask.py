import os
import sys

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as std_trnsf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.networks import get_network

DEVICE = 'cpu'

def prepare_network():
    network = 'pspnet_resnet101'

    net = get_network(network).to(DEVICE)
    state = torch.load('./models/pspnet_resnet101_adam_lr_0.0001_epoch_47.pth',
                       map_location='cpu')
    net.load_state_dict(state['weight'])

    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return test_image_transforms, net


def prepare_hair_mask(img_path, test_image_transforms, net):
    img = Image.open(img_path)

    data = test_image_transforms(img)
    data = torch.unsqueeze(data, dim=0)
    net.eval()
    data = data.to(DEVICE)

    logit = net(data)

    pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
    pil_mask = pred >= 0.5

    hair_mask = np.zeros(pil_mask.shape[:2], dtype=np.uint8)
    hair_mask[:] = 255 * pil_mask

    return hair_mask
