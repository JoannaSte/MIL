import lightning as pl
import torch
import torch.nn as nn
from typing import Any

class Learner(pl.LightningModule):
    def __init__(self,
                 model,
                 learning_rate: float = 0.0001,
                 weight_decay: float = 0.0001,
                 *args: Any,
                 **kwargs: Any,
                 ) -> None:
         super().__init__(*args,**kwargs)
         
         self.model = model
         self.loss = nn.L1Loss()
         self.learning_rate = learning_rate
         self.weight_decay=weight_decay
         
    def _step(self, batch):
        
        data, label = batch
        x = self.model(data)
        x = x.squeeze().float()
        label = label.squeeze().float()
        loss = self.loss(x,label)
        
        return loss
    
    def training_step(self, batch):
        
        loss=self._step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
     
    def validation_step(self,batch):
        
        with torch.no_grad():
            
            loss = self._step(batch)        
            metrics = {"validation_loss": loss}
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        return metrics
    
    def configure_optimizers(self):
        
        optimizer= torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        
        return optimizer
    
    def forward(self, x):
        
        y = self.model(x)
        
        return y