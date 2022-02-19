import os
import pickle
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T

modelName = "./models/19-02-22-03_03.pth"
resultsPath = "../../data/results/maskrcnn_raw"
datasetPath = "../../annotations/maskrcnn/evaluation/imgs"

class Dataset(object):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        loader = T.Compose([T.ToTensor()])
        img = loader(img, None)[0]
        img = img.cuda()

        return img

    def __len__(self):
        return len(self.imgs)

def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    loader = T.Compose([T.ToTensor()])
    image = loader(image, None)[0]
    #image = image.unsqueeze(0)
    if torch.cuda.is_available():
        return [image.cuda()]
    else:
        return [image]

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torch.load(modelName, map_location=device)
    model.to(device)
    model.eval()

    outputFolder = os.path.join(resultsPath, modelName.split("/")[-1])
    os.makedirs(outputFolder, exist_ok=True)

    imgs = list(os.listdir(datasetPath))
    for imgfn in imgs:
        path = os.path.join(datasetPath, imgfn)
        img = image_loader(path)
        result = model(img)

        masks = []
        for idx in range(len(result[0]["masks"])):
            masks.append(result[0]["masks"][idx][0].detach().cpu().numpy())
        result[0]["masks"] = masks

        resultfn = os.path.join(resultsPath, modelName.split("/")[-1], imgfn + ".pickle")
        with open(resultfn, "wb") as f:
            pickle.dump(result, f, protocol=2)

    """
    img = np.array(result[0]["masks"][0][0].detach().cpu().numpy())
    for row in range(len(img)):
        for val in range(len(img)):
            img[row][val] = (img[row][val])*255
    cv2.imwrite("et.png", img)
    """

if __name__ == "__main__":
    main()
