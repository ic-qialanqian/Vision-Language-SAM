# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from PIL import Image
from torchvision import transforms
from typing import Any, Optional, Tuple, Type
from transformers import AutoModel, AutoTokenizer
from utils_downstream.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc
import torch
# set seeds
torch.manual_seed(2024)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

def eval_psnr(loader, model,eval_type=None,device=None):
    model.eval()


    
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    
    mae,sm,em,wfm, m_dice, m_iou,ber,acc= cal_mae(),cal_sm(),cal_em(),cal_wfm(), cal_dice(), cal_iou(),cal_ber(),cal_acc()
    #save_path = './work_dir/..../'
    #os.makedirs(save_path, exist_ok=True)
    pred_list = []
    gt_list = []
    #for batch in loader:
    for step, (image, gt2D,img_path) in enumerate(loader):
        image, gt2D = image.to(device), gt2D.to(device)
        
        with torch.no_grad():
            pred = torch.sigmoid(model(image))
            
            res = pred.squeeze().squeeze().cpu().numpy()
            gt = gt2D.squeeze().squeeze().cpu().numpy()

            mae.update(res, gt)
            sm.update(res,gt)
            #fm.update(res, gt)
            em.update(res,gt)
            wfm.update(res,gt)
            m_dice.update(res,gt)
            m_iou.update(res,gt)
            ber.update(res,gt)
            
            '''
            seg_prob = pred.cpu().numpy().squeeze()
            seg_prob = (seg_prob > 0.5).astype(np.uint8)* 255
            img_name = img_path[0].split('/')[-1]
            img = Image.fromarray(seg_prob)
            img.save(os.path.join(save_path, img_name))
            '''
        if pbar is not None:
            pbar.update(1)

    MAE = mae.show()
    #maxf,meanf,_,_ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    ber = ber.show()

    if pbar is not None:
        pbar.close()

    return sm, em, wfm, MAE
    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "GT/")
        self.img_path = join(data_root, "Imgs/")
        self.gt_path_files = sorted([self.gt_path + f for f in os.listdir(self.gt_path) if f.endswith('.png')])
        self.img_path_files = sorted([self.img_path + f for f in os.listdir(self.img_path) if f.endswith('.jpg') or f.endswith('.png')])
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")
        
        self.img_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):

        img_1024 = Image.open(self.img_path_files[index]).convert('RGB')
        
        gt = Image.open(self.gt_path_files[index]).convert('L')  # multiple labels [0, 1,4,5...], (256,256)
        
        img_1024 = self.img_transform(img_1024)
        gt = self.mask_transform(gt)
        

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt).long(),
            self.img_path_files[index]
        )


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/TrainDataset",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_h")
parser.add_argument(
    "-checkpoint", type=str, default="..../vlsam_model_best.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model

        
class VLSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder

        self.pe_layer = PositionEmbeddingRandom(256 // 2)
        self.no_mask_embed = nn.Embedding(1, 256)
        
    def forward(self, image):

        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)   
      
        bs,c,h,w = image_embedding.shape
        sparse_embeddings = torch.empty((bs, 0, 256),device=image.device)       
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, 64, 64
            )
        
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.pe_layer((64,64)).unsqueeze(0),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():
    sam_model = sam_model_registry[args.model_type](checkpoint='work_dir/SAM/sam_vit_h_4b8939.pth')
    vlsam_model = VLSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    vlsam_model.load_state_dict(checkpoint["model"])

    
    test_dataset = NpyDataset('data/TestDataset/CHAMELEON')
    print("Number of training samples: ", len(test_dataset))

    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
              
        
    result1, result2, result3, result4 = eval_psnr(test_dataloader, vlsam_model,
                eval_type='cod',device=device)
    print('Sm', {'val': result1})
    print('Em', {'val': result2})
    print('Fm', {'val': result3})
    print('MAE', {'val': result4})




if __name__ == "__main__":
    main()
