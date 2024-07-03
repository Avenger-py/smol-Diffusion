# smol-Diffusion
Implemented DDPM Diffusion model from scratch. Used a Unet model with Self-attention.

**Generated Images**

<img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/generated_img1.png"> <img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/generated_img2.png">

<img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/generated_img3.png"> <img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/generated_img4.png">

## Dataset
I used Stanford Cars Dataset, which contains about 16k car images. Steps to download dataset: https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616

## Requirements
### Software
- torch
- torchvision
- scipy
- numpy
- matplotlib
- tqdm
- Jupyterlab

### Hardware
GPU is absolutely required for training and sampling.
I used Runpod cloud gpu.

## How to run?
1. Clone the repo
2. Install the requirements mentioned above
3. Place the dataset inside `data` folder in the working directory
4. Run the notebook `DDPM_train_and_sample.ipynb`

## Training
Total training took around 40 hrs on a single GPU with 16gb VRAM and a batch size of 32.
I trained it for roughly 380 epochs (1 epoch = 1 complete pass through the dataset = 16k imgs / 32 batch size = 500 iters of 32 batch size).

## Sampling
Sampling is quite slow. I used 15000 timesteps to obtain respectable results.
Sampling is major limitation of DDPM diffusion. The images are blurry, less sharp, less colorful and look like mean of the data.

## Sampling results
<img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/img_header.png">
<img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/results6.png">
<img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/results1.png">
<img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/results2.png">
<img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/results3.png">
<img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/results4.png">
<img src="https://github.com/Avenger-py/smol-Diffusion/blob/main/assets/results5.png">

## Resources and Acknowledgments
Based on the paper: *Denoising Diffusion Probabilistic Models* https://arxiv.org/abs/2006.11239

Inspired by these youtube videos: 
- https://www.youtube.com/watch?v=a4Yfz2FxXiY&ab_channel=DeepFindr
- https://www.youtube.com/watch?v=HoKDTa5jHvg&ab_channel=Outlier

Great blog post on diffusion: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

Other useful resources:
- https://github.com/openai/improved-diffusion/tree/main
- https://arxiv.org/abs/2102.09672
