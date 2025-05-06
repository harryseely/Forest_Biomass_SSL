
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import R2Score

from utils.data_utils import convert_z_to_mg_ha, update_z_score_conversion_info
from models.ocnn_lenet import OCNN_LeNet

class OCNNLightning(pl.LightningModule):
    """
    *** IMPORTANT NOTE: The target biomass tensor order is [foliage, bark, branch, wood]
    """
    def __init__(self, cfg, z_info=None, predict_output="pred_and_target"):
        super().__init__()

        self.cfg = cfg
        self.predict_output = predict_output

        if cfg['task'] == "pretrain":
            target_len = len(cfg['target_lidar_metrics'])
        else:
            target_len = len(cfg['biomass_comp_nms'])
            
            if z_info is None:
                self.z_info = update_z_score_conversion_info(
                    biomass_labels_fpath=cfg['labels_rel_fpath'],
                    biomass_comp_nms=cfg['biomass_comp_nms'])
            else:
                self.z_info = z_info

        if self.cfg['verbose']:
            print(f"\nTraining OCNN with a target of length: {target_len}\n")

        if self.cfg['model_nm'] == "ocnn_lenet":

          self.model = OCNN_LeNet(cfg=self.cfg,
                            in_channels=3,
                            out_channels=target_len,
                            ocnn_stages=cfg['ocnn_stages'],
                            ocnn_late_channels=64,
                            dropout=self.cfg['dropout'],
                            nempty=cfg['octree_nempty'])
          
        else:
            raise ValueError(f"Invalid model name ({cfg['model_nm']}) specified in config file.")

        if self.cfg['loss_fn'] == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        elif self.cfg['loss_fn'] == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss function ({cfg['loss_fn']}) specified in config file.")

        self.train_r2 = R2Score(adjusted=0, multioutput='uniform_average', dist_sync_on_step=False)

        self.val_r2 = R2Score(adjusted=0, multioutput='uniform_average', dist_sync_on_step=False)

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)

        train_loss = self.loss_fn(input=pred, target=batch['target'])

        self.log("train_loss", value=train_loss, batch_size=self.cfg['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_r2(pred, batch['target'])
        self.log("train_r2", self.train_r2, on_epoch=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)

        val_loss = self.loss_fn(input=pred, target=batch['target'])

        self.log("val_loss", value=val_loss, batch_size=self.cfg['batch_size'],
                 on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_r2(pred, batch['target'])
        self.log("val_r2", self.val_r2, on_epoch=True)

        return val_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        pred = self.model(batch)

        pred = convert_z_to_mg_ha(z_info=self.z_info,
                            z_components_arr=pred,
                            biomass_comp_nms=self.cfg['biomass_comp_nms'])

        if self.predict_output == "pred_and_target":
            target_or_cellid = convert_z_to_mg_ha(z_info=self.z_info,
                                        z_components_arr=batch['target'],
                                        biomass_comp_nms=self.cfg['biomass_comp_nms'])
        
        elif self.predict_output == "pred_and_cellid":
            target_or_cellid = batch['cell_id']

        else:
            raise ValueError(f"Invalid prediction output ({self.predict_output}) specified in config file.")

        return pred, target_or_cellid

    def configure_optimizers(self):
        assert self.trainer is not None

        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.cfg['lr'],  
            weight_decay=self.cfg['weight_decay'],
        )

        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
                                                                     T_0=self.cfg['n_epochs'],
                                                                     T_mult=2
                                                                     )

        return [opt], [sched]

    def load_checkpoint(self, path: str) -> None:
        
        print(f"\033[92mLoading checkpoint from '{path}'.\033[0m")
        
        #Load the pytorch lightning checkpoint dict, only keeping the state_dict item
        checkpoint = torch.load(path)['state_dict']

        #Drop regressor module (different number of outputs leads to mismatch)
        checkpoint = {k: v for k, v in checkpoint.items() if "regressor" not in k}

        #Load the state dict weights and biases to OCNN
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)  # type: ignore

        #Check that only missing keys were from the regressor module
        missing_keys = [k for k in missing_keys if "regressor" not in k]

        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        assert (len(missing_keys) == 0 & len(unexpected_keys) == 0), \
            f"OCNN checkpoint missing the keys and/or has unexpected keys"

