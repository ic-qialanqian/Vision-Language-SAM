# Vision-Language SAM


## Installation
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/bowang-lab/MedSAM`
4. Enter the SAM folder `cd SAM` and run `pip install -e .`



## Model Training

### Data preprocessing

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at `work_dir/SAM/sam_vit_b_01ec64.pth` .


### Training on one GPU

```bash
python train_sam_blip_mamba.py
```

### Inference

```bash
python Inference_blip_mamba.py
```




## Reference


