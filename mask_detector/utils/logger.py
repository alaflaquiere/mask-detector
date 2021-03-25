from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, logdir_p: Path):
        self.logdir_p = logdir_p
        self.logdir_p.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.logdir_p)
        self.epoch = 1

    def update(self, *dictionaries):
        for d in dictionaries:
            for key, value in d.items():
                self.writer.add_scalar(key, value, self.epoch)
        self.epoch += 1

    def set_epoch(self, epoch):
        self.epoch = epoch
