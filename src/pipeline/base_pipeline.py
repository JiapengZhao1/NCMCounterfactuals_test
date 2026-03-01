import pytorch_lightning as pl
import torch as T
from torch.utils.data import DataLoader, Dataset


class BasePipeline(pl.LightningModule):
    min_delta = 1e-5
    patience = 20
    max_epochs = 10000

    # Exponential moving average of the monitored training loss.
    # Smaller alpha => smoother curve.
    loss_ema_alpha = 0.05

    def __init__(self, generator, do_var_list, dat_sets, cg, dim, ncm, batch_size=256):
        super().__init__()
        self.generator = generator
        self.do_var_list = do_var_list
        self.dat_sets = dat_sets
        self.cg = cg
        self.ncm = ncm
        self.dim = dim

        self.batch_size = batch_size

        self.stored_metrics = None

        # EMA state (python float is enough)
        self._train_loss_ema = None

    def forward(self, n=1, u=None, do={}):
        return self.ncm(n, u, do)

    def train_dataloader(self):
        return DataLoader(SCMDataset(self.dat_sets),
            batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

    def update_metrics(self, new_metrics):
        self.stored_metrics = new_metrics

    def on_train_epoch_end(self):
        """Log an EMA-smoothed version of train_loss for stable checkpointing/early-stopping."""
        # pipelines in this repo log 'train_loss' each epoch; read it back from callback metrics
        current = self.trainer.callback_metrics.get('train_loss', None) if getattr(self, 'trainer', None) else None
        if current is None:
            return
        try:
            cur_val = float(current.detach().cpu()) if hasattr(current, 'detach') else float(current)
        except Exception:
            return

        if self._train_loss_ema is None:
            self._train_loss_ema = cur_val
        else:
            a = float(getattr(self, 'loss_ema_alpha', 0.05))
            self._train_loss_ema = (1.0 - a) * self._train_loss_ema + a * cur_val

        # make it visible to callbacks
        self.log('train_loss_ema', self._train_loss_ema, prog_bar=True)


class SCMDataset(Dataset):
    def __init__(self, dat_sets):
        self.dat_sets = dat_sets
        self.length = len(dat_sets[0][next(iter(dat_sets[0]))])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [{k: self.dat_sets[i][k][idx] for k in self.dat_sets[i]} for i in range(len(self.dat_sets))]
