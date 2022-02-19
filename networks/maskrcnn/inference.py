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

modelName = "./models/19-02-22-00_44.pth"
resultsPath = "../../results/maskrcnn/"
datasetPath = "../../"

class Dataset(object):
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

def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    loader = T.Compose([T.ToTensor()])
    image = loader(image, None)[0]
    #image = image.unsqueeze(0)
    return [image.cuda()]

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torch.load(modelName)
    model.to(device)
    model.eval()

    img = image_loader("test/2020-11-17-13-47-41.bag.pickle-0.png")

    #dataset = Dataset(datasetPath, get_transform(train=False))

    out = model(img)
    with open("dump", "wb") as f:
        pickle.dump(out, f)

    img = np.array(out[0]["masks"][0][0].detach().cpu().numpy())
    for row in range(len(img)):
        for val in range(len(img)):
            img[row][val] = (img[row][val])*255
    cv2.imwrite("et.png", img)

if __name__ == "__main__":
    main()
