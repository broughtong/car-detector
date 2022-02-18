import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

datasetPath = "./test"

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        #if self.transforms is not None:
        #    img, target = self.transforms(img, target)

        return img

    def __len__(self):
        return len(self.imgs)

#loader = T.Compose([T.Scale(imsize), T.ToTensor()])
floader = T.Compose([T.ToTensor()])

def timage_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert("RGB")
    image = floader(image, None)[0]
    print(image)
    print(image.shape)

    #image = image.unsqueeze(0)
    return [image.cuda()]

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torch.load("models/model")

    # move model to the right device
    model.to(device)
    model.eval()

    img = timage_loader("test/test.png")

    
    #dataset = PennFudanDataset(datasetPath, get_transform(train=False))
    #img = Image.open("test/test.png").convert("RGB")

    out = model(img)
    import pickle
    with open("dump", "wb") as f:
        pickle.dump(out, f)

    img = np.array(out[0]["masks"][0][0].detach().cpu().numpy())
    for row in range(len(img)):
        for val in range(len(img)):
            img[row][val] = (img[row][val])*255
    import cv2
    cv2.imwrite("et.png", img)

    #img = np.array(d[0]["masks"][0][0].detach().cpu().numpy())
    
if __name__ == "__main__":
    main()
