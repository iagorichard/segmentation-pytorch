from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch import optim
import os, shutil
from IPython.display import clear_output
from sklearn.metrics import jaccard_score as iou
import torch.nn.functional as F
import numpy as np
import time

from src.dice import dice_coeff
from src.dataset_pytorch import BasicDataset
from src.general import get_checkpoint_folders
        

class Train:

    def __init__(self, model, device, epochs, bs, dataset_info, infos):
        self.model_name, self.model = next(iter(model.items()))
        self.device       = device
        self.epochs       = epochs
        self.bs           = bs
        self.dataset_info = dataset_info
        self.infos     = infos



###################################################################

    def __get_dataloaders__(self):
        
        dataset = BasicDataset(self.img_dir, self.mask_dir)
        
        n_val      = int(len(dataset) * self.val_size)
        n_train    = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        
        train_loader = DataLoader(train, batch_size=self.bs, shuffle=False, num_workers=14, pin_memory=True)
        val_loader   = DataLoader(val,   batch_size=self.bs, shuffle=False, num_workers=14, pin_memory=True)

        return train_loader, val_loader, n_train, n_val

###################################################################

    def __get_global_model_params__(self):
        
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if self.model.segmentation_head[0].out_channels > 1 else 'max', patience=2)
        
        '''
        if self.model.segmentation_head[0].out_channels > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        '''
        #considering multiple classes for segmentation
        criterion = nn.CrossEntropyLoss()

        return optimizer, criterion

###################################################################

    def __delete_previous_checkpoints__(self, folder):
        
        file_list = os.listdir(folder)

        # Itere sobre cada arquivo
        for file_name in file_list:
            # Verifique se o arquivo tem a extensão .pth
            if file_name.endswith(".pth"):
                # Construa o caminho completo do arquivo
                file_path = os.path.join(folder, file_name)
                # Remova o arquivo
                os.remove(file_path)

###################################################################
                
    def __transform__(self, tensor):
        return tensor.cpu().numpy().flatten()


    def model_evaluation(self, model, loader, device):
        """Evaluation without the densecrf with the dice coefficient"""
        model.eval()
        #mask_type = torch.float32 if model.segmentation_head[0].out_channels == 1 else torch.long
        #assuming it's semantic segmentation
        mask_type = torch.long
        n_val = len(loader)  # the number of batch
        tot_iou = 0
        tot_dice = 0
        tot = 0

        #for batch in tqdm(loader):
        for batch in tqdm(loader):
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = model(imgs)

            #if model.segmentation_head[0].out_channels > 1:
            if True:
                true_masks = torch.argmax(true_masks, 1)
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                
                tot_dice += dice_coeff(pred, true_masks).item()
                tot_iou += iou(self.__transform__(true_masks), self.__transform__(pred))

        model.train()
        return tot / n_val

###################################################################

    def fit(self):

        train_loader, val_loader = self.dataset_info
        optimizer, criterion = self.__get_global_model_params__()
        
        val_loss_lst = []
        train_loss_lst = []
        best_loss = 3200000000000
        
        epoch = 0
        ckpt_base, ckpt_last_folder, ckpt_best_folder = get_checkpoint_folders(self.infos, self.model_name)

        for epoch in range(self.epochs):
            clear_output(wait=False)
            print(f"Training {self.model_name}...")
            self.model.train()

            epoch_loss = 0
            n_batches = len(train_loader)  # Assumindo que train_loader pode fornecer o número total de batches
            with tqdm(total=n_batches, desc=f'Epoch {epoch + 1}/{self.epochs}', unit='batch') as pbar:
                for batch_idx, batch in enumerate(train_loader, start=1):
                    imgs = batch['image']
                    true_masks = batch['mask']

                    imgs = imgs.to(device=self.device, dtype=torch.float32)
                    mask_type = torch.long
                    true_masks = true_masks.to(device=self.device, dtype=mask_type)

                    masks_pred = self.model(imgs)
                    true_masks = torch.argmax(true_masks, 1)

                    loss = criterion(masks_pred, true_masks)
                    epoch_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    
                    nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    optimizer.step()
                    
                    pbar.set_postfix(loss=epoch_loss / batch_idx)  # Opcional: mostra o loss médio até agora
                    pbar.update(1)  # Atualiza a barra de progresso em 1

            clear_output(wait=False)
            print(f"Evaluating {self.model_name} after epoch {epoch+1}...")
            
            val_lss = self.model_evaluation(self.model, val_loader, self.device)
            print(f"Validation result is: {val_lss}")
            
            val_loss_lst.append(val_lss)
            train_loss_lst.append(loss.cpu().item())

            if val_lss < best_loss:
                best_loss = val_lss

                self.__delete_previous_checkpoints__(ckpt_best_folder)
                cp_file = ckpt_best_folder + f'/BEST_CP_epoch{epoch + 1}.pth'
                torch.save(self.model.state_dict(), cp_file)
                print("Checkpoint saved!")

            #put a sleep to view the epoch result
            time.sleep(3)

        #save last checkpoint
        cp_file = ckpt_last_folder + f'/LAST_CP_epoch{epoch + 1}.pth'
        torch.save(self.model.state_dict(), cp_file)
        
        #save all losses in all epochs
        np.savetxt(ckpt_base+'/LOSSES_TRAIN.csv', np.array(train_loss_lst), fmt='%.6f')
        np.savetxt(ckpt_base+'/LOSSES_VAL.csv', np.array(val_loss_lst), fmt='%.6f')