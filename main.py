import os

import cv2
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet


class PictureDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None, target_transform=None):
        self.imgs = []
        count = 0
        for root, dirs, files in os.walk(x_path):
            for file in files:
                self.imgs.append(str(file))
                count += 1
        self.x_dir_path = x_path
        self.y_dir_path = y_path
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = y_path = self.imgs[index]
        img_x = cv2.imread(os.path.join(self.x_dir_path, x_path))
        # img_x = np.rot90(img_x, 1)
        # img_x = Image.fromarray(img_x)
        # img_x = Image.fromarray(cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB))
        img_x = Image.open(os.path.join(self.x_dir_path, x_path))
        img_y = Image.open(os.path.join(self.y_dir_path, y_path))

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
class TestPictureDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None, target_transform=None):
        self.imgs = []
        count = 0
        for root, dirs, files in os.walk(x_path):
            for file in files:
                self.imgs.append(str(file))
                count += 1
        self.x_dir_path = x_path
        self.y_dir_path = y_path
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = y_path = self.imgs[index]
        img_x = cv2.imread(os.path.join(self.x_dir_path, x_path))
        # img_x = np.rot90(img_x, 1)
        # img_x = Image.fromarray(img_x)
        # img_x = Image.fromarray(cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB))
        img_x = Image.open(os.path.join(self.x_dir_path, x_path))
        img_y = Image.open(os.path.join(self.y_dir_path, y_path))

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y,x_path

    def __len__(self):
        return len(self.imgs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_transforms = transforms.Compose([
    transforms.Resize((512, 512)),

    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_transforms_y = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            # exit()
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
        torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model



def train(batch_size):
    model = Unet(3, 3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    dataset = PictureDataset("dataset/x", "dataset/y", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


def test(model_path):
    model = Unet(3, 3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    test_dataset = TestPictureDataset("dataset/x", "dataset/y", transform=test_transforms,
                                  target_transform=test_transforms_y)
    dataloaders = DataLoader(test_dataset, batch_size=1)

    model.eval()
    plt.ion()
    with torch.no_grad():
        for x, y_raw,filename in dataloaders:
            y = model(x)

            img_y = torch.squeeze(y)

            # img_y = transforms.ToPILImage(img_y)
            # img_y = np.transpose(img_y,(1,2,0))
            # print(img_y)

            # img_y.resize((3376,6000,3))
            # print(img_y.shape)
            # cv2.imshow("res",img_y)
            # cv2.waitKey()

            unloader = transforms.ToPILImage()
            image = unloader(img_y)
            image = np.asarray(image)
            # BGR
            # RGB
            # image = image[:,:,0,1,2]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (y_raw.numpy().shape[-1], y_raw.numpy().shape[-2]), interpolation=cv2.INTER_AREA)
            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(image)
            # plt.subplot(1, 3, 2)
            # plt.imshow(image)
            # plt.title("3")
            # plt.subplot(1, 3, 3)
            # plt.imshow(image)
            # plt.show()
            cv2.resizeWindow("resized", 640, 480);
            cv2.imshow("Result", image)
            cv2.imshow("Raw", cv2.imread(os.path.join("./dataset/x",filename[0])))
            cv2.imshow("True", cv2.imread(os.path.join("./dataset/y",filename[0])))
            cv2.waitKey()


def test_single_picture(model_path, x_path, transform):
    model = Unet(3, 3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        x = Image.open(x_path)
        print(x.size)
        raw_width = x.size[0]
        raw_height = x.size[1]
        x = transform(x)
        x = torch.tensor(np.array([x.numpy()]))
        print(x.numpy().shape)
        y = model(x)
        img_y = torch.squeeze(y)

        unloader = transforms.ToPILImage()
        image = unloader(img_y)
        image = np.asarray(image)
        # RGB BGR
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (raw_width, raw_height), interpolation=cv2.INTER_LANCZOS4)

        cv2.imshow("Result", image)
        cv2.imshow("Raw", cv2.imread(x_path))
        cv2.waitKey(0)


if __name__ == '__main__':
    # test_single_picture("./weights_19.pth","C:\\Users\\STVEA\\Pictures\\6.jpg",test_transforms)
    # test("./1.pth")
    test("./weights_19.pth")
    # train(2)
