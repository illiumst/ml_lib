from collections import defaultdict

from abc import ABC
from argparse import Namespace

import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from torchvision.transforms import Compose

from ml_lib._templates.new_project.datasets.template_dataset import TemplateDataset

from ml_lib.audio_toolset.audio_io import NormalizeLocal
from ml_lib.modules.util import LightningBaseModule
from ml_lib.utils.transforms import ToTensor

from ml_lib._templates.new_project.utils.project_config import GlobalVar as GlobalVars


class BaseOptimizerMixin:

    def configure_optimizers(self):
        assert isinstance(self, LightningBaseModule)
        opt = Adam(params=self.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        if self.params.sto_weight_avg:
            # TODO: Make this glabaly available.
            opt = SWA(opt, swa_start=10, swa_freq=5, swa_lr=0.05)
        return opt

    def on_train_end(self):
        assert isinstance(self, LightningBaseModule)
        for opt in self.trainer.optimizers:
            if isinstance(opt, SWA):
                opt.swap_swa_sgd()

    def on_epoch_end(self):
        assert isinstance(self, LightningBaseModule)
        if self.params.opt_reset_interval:
            if self.current_epoch % self.params.opt_reset_interval == 0:
                for opt in self.trainer.optimizers:
                    opt.state = defaultdict(dict)


class BaseTrainMixin:

    absolute_loss = nn.L1Loss()
    nll_loss = nn.NLLLoss()
    bce_loss = nn.BCELoss()

    def training_step(self, batch_xy, batch_nb, *_, **__):
        assert isinstance(self, LightningBaseModule)
        batch_x, batch_y = batch_xy
        y = self(batch_x).main_out
        bce_loss = self.bce_loss(y, batch_y)
        return dict(loss=bce_loss, log=dict(batch_nb=batch_nb))

    def training_epoch_end(self, outputs):
        assert isinstance(self, LightningBaseModule)
        keys = list(outputs[0].keys())

        summary_dict = {f'mean_{key}': torch.mean(torch.stack([output[key]
                                                               for output in outputs]))
                        for key in keys if 'loss' in key}
        for key in summary_dict.keys():
            self.log(key, summary_dict[key])


class BaseValMixin:

    absolute_loss = nn.L1Loss()
    nll_loss = nn.NLLLoss()
    bce_loss = nn.BCELoss()

    def validation_step(self, batch_xy, batch_idx, _, *__, **___):
        assert isinstance(self, LightningBaseModule)
        batch_x, batch_y = batch_xy
        y = self(batch_x).main_out
        val_bce_loss = self.bce_loss(y, batch_y)
        return dict(val_bce_loss=val_bce_loss,
                    batch_idx=batch_idx, y=y, batch_y=batch_y)

    def validation_epoch_end(self, outputs, *_, **__):
        assert isinstance(self, LightningBaseModule)
        summary_dict = dict()
        # In case of Multiple given dataloader this will outputs will be: list[list[dict[]]]
        # for output_idx, output in enumerate(outputs):
        # else:list[dict[]]
        keys = list(outputs.keys())
        # Add Every Value das has a "loss" in it, by calc. mean over all occurences.
        summary_dict.update({f'mean_{key}': torch.mean(torch.stack([output[key]
                                                                    for output in outputs]))
                             for key in keys if 'loss' in key}
                            )
        """
        # Additional Score like the unweighted Average Recall:
        # UnweightedAverageRecall
        y_true = torch.cat([output['batch_y'] for output in outputs]) .cpu().numpy()
        y_pred = torch.cat([output['y'] for output in outputs]).squeeze().cpu().numpy()

        y_pred = (y_pred >= 0.5).astype(np.float32)

        uar_score = sklearn.metrics.recall_score(y_true, y_pred, labels=[0, 1], average='macro',
                                                 sample_weight=None, zero_division='warn')

        summary_dict['log'].update({f'uar_score': uar_score})
        """

        for key in summary_dict.keys():
            self.log(key, summary_dict[key])


class BinaryMaskDatasetMixin:

    def build_dataset(self):
        assert isinstance(self, LightningBaseModule)

        # Dataset
        # =============================================================================
        # Data Augmentations or Utility Transformations

        transforms = Compose([NormalizeLocal(), ToTensor()])

        # Dataset
        dataset = Namespace(
            **dict(
                # TRAIN DATASET
                train_dataset=TemplateDataset(self.params.root, setting=GlobalVars.DATA_OPTIONS.train,
                                              transforms=transforms
                                              ),

                # VALIDATION DATASET
                val_dataset=TemplateDataset(self.params.root, setting=GlobalVars.vali,
                                            ),

                # TEST DATASET
                test_dataset=TemplateDataset(self.params.root, setting=GlobalVars.test,
                                             ),

            )
        )
        return dataset


class BaseDataloadersMixin(ABC):

    # Dataloaders
    # ================================================================================
    # Train Dataloader
    def train_dataloader(self):
        assert isinstance(self, LightningBaseModule)
        # In case you want to implement bootstraping
        # sampler = RandomSampler(self.dataset.train_dataset, True, len(self.dataset.train_dataset))
        sampler = None
        return DataLoader(dataset=self.dataset.train_dataset, shuffle=True if not sampler else None, sampler=sampler,
                          batch_size=self.params.batch_size,
                          num_workers=self.params.worker)

    # Test Dataloader
    def test_dataloader(self):
        assert isinstance(self, LightningBaseModule)
        return DataLoader(dataset=self.dataset.test_dataset, shuffle=False,
                          batch_size=self.params.batch_size,
                          num_workers=self.params.worker)

    # Validation Dataloader
    def val_dataloader(self):
        assert isinstance(self, LightningBaseModule)
        val_dataloader = DataLoader(dataset=self.dataset.val_dataset, shuffle=False,
                                    batch_size=self.params.batch_size, num_workers=self.params.worker)
        # Alternative return [val_dataloader, alternative dataloader], there will be a dataloader_idx in validation_step
        return val_dataloader
