import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io
from pandas import DataFrame


class MaskDataset(Dataset):

    def __init__(self, dataFrame: DataFrame):
        self.df = dataFrame
        self.transforms = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((100, 100)),
             transforms.ToTensor()]
        )

    def __getitem__(self, i: int):
        img_path, mask = self.df.iloc[i]
        img = io.imread(img_path)
        img = self.transforms(img)
        mask = torch.tensor(mask, dtype=torch.long)
        return {"image": img, "mask": mask}

    def __len__(self):
        return self.df.shape[0]

    def explore(self, k: int = 12):
        rnd_indexes = torch.randperm(self.df.shape[0])[:k]
        samples = [self[int(i)]["image"] for i in rnd_indexes]
        samples = map(lambda x: x.numpy().transpose(1, 2, 0), samples)
        io.imshow_collection(list(samples))
        io.show()
