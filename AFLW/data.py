import json
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image


class AFLWDataset(Dataset):

    def __init__(self, training=True):
        self.root_url = "/kaggle/input/aflw-data/aflw/images/"
        self.train_url = "/kaggle/input/aflw-data/aflw/annotations/face_landmarks_aflw_train.json"
        self.test_url = "/kaggle/input/aflw-data/aflw/annotations/face_landmarks_aflw_test.json"

        self.transform = transforms.Compose([
                                transforms.RandomResizedCrop((256, 256), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if training:
            with open(self.train_url) as train:
                raw_train_data = json.load(train)
            self.data = [datum for datum in raw_train_data['images']]
        else:
            with open(self.test_url) as test:
                raw_test_data = json.load(test)
            self.data = [datum for datum in raw_test_data['images']]

    def __getitem__(self, index):
        img_url = os.path.join(self.root_url, self.data[index]['file_name'])
        with open(img_url, "rb") as f:
            img = Image.open(f)
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img

    def __len__(self):
        return len(self.data)
