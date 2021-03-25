from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.logger import Logger


class MaskClassifier(nn.Module):

    def __init__(self, train_dataset: Dataset = None, val_dataset: Dataset = None,
                 batch_size: int = 32, n_epochs: int = 10, device: str = "cuda",
                 save_path: Path = Path("checkpoints"), logger: Logger = None):
        super().__init__()
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.logger = logger
        self.save_p = save_path

        # datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # dataloaders
        if train_dataset:
            self.train_dataloader = DataLoader(train_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=2)
        if val_dataset:
            self.val_dataloader = DataLoader(val_dataset,
                                             batch_size=self.batch_size,
                                             num_workers=2)

        # NNet
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3),
                      padding=(1, 1), stride=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

        # enforce Kaiming initialization
        for sequential in [self.convlayer1, self.convlayer2, self.convlayer3, self.head]:
            for layer in sequential.children():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_uniform_(layer.weight)

        # class weighting to counter class imbalance
        if self.train_dataset:
            n_masked = sum(self.train_dataset.df["mask"] == 1)
            n_nonmasked = sum(self.train_dataset.df["mask"] == 0)
            self.class_weigthing = 1 - \
                torch.tensor([n_masked, n_nonmasked],
                             dtype=torch.float) / (n_masked + n_nonmasked)
        else:
            self.class_weigthing = torch.tensor([1, 1], dtype=torch.float)

        # weighted loss function
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weigthing)

        # optimizer
        self.optimizer = Adam(self.parameters(),
                              lr=0.0001)

        # move module parameters to device
        self.to(self.device)

    def forward(self, x):
        y = self.convlayer1(x)
        y = self.convlayer2(y)
        y = self.convlayer3(y)
        y = y.view(-1, 2048)
        y = self.head(y)
        return y

    def training_step(self, batch):
        images, labels = batch["image"].to(
            self.device), batch["mask"].to(self.device)
        self.train()
        self.optimizer.zero_grad()
        outputs = self.forward(images)
        loss_batch = self.loss_fn(outputs, labels)
        loss_batch.backward()
        self.optimizer.step()
        return {"loss_batch": loss_batch.item()}

    def training_epoch(self, epoch):
        train_loss = 0.
        for batch in tqdm(self.train_dataloader, desc="epoch {} - train".format(epoch)):
            res = self.training_step(batch)
            train_loss += res["loss_batch"]
        train_loss = train_loss / len(self.train_dataloader)
        return {"train_loss": train_loss}

    def validation_step(self, batch):
        images, labels = batch["image"].to(
            self.device), batch["mask"].to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)
            loss_batch = self.loss_fn(outputs, labels)
            _, classes = torch.max(outputs, dim=1)
            accuracy = accuracy_score(classes.cpu(), labels.flatten().cpu())
        return {"val_loss_batch": loss_batch.item(), "accuracy": accuracy}

    def validation_epoch(self, epoch):
        val_loss = 0.
        val_accuracy = 0.
        for batch in tqdm(self.val_dataloader, desc="epoch {} - val".format(epoch)):
            res = self.validation_step(batch)
            val_loss += res["val_loss_batch"]
            val_accuracy += res["accuracy"]
        val_loss = val_loss / len(self.val_dataloader)
        val_accuracy = val_accuracy / len(self.val_dataloader)
        return {"val_loss": val_loss, "val_accuracy": val_accuracy}

    def fit(self):
        best_solution = {"epoch": 0, "val_accuracy": 0.}
        for epoch in range(self.n_epochs):
            res_train = self.training_epoch(epoch)
            res_val = self.validation_epoch(epoch)
            if self.logger is not None:
                self.logger.update(res_train, res_val)
            self.save(epoch)
            print("train loss: {:1.3E} | val loss: {:1.3E} - val acc: {}".format(res_train["train_loss"],
                                                                                 res_val["val_loss"],
                                                                                 res_val["val_accuracy"]))
            if res_val["val_accuracy"] > best_solution["val_accuracy"]:
                best_solution["epoch"] = epoch
                best_solution["val_accuracy"] = res_val["val_accuracy"]
        return best_solution

    def save(self, epoch: int):
        self.save_p.mkdir(parents=True, exist_ok=True)
        model_name = "model_epoch{:03}.ckpt".format(epoch)
        torch.save(self.state_dict(), self.save_p / model_name)
