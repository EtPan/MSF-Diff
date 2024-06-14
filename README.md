# From the Abundance Perspective: Multi-modal Scene Fusion-based Hyperspectral Image Synthesis



## Introduction

This is the source code for our paper: [From the Abundance Perspective: Multi-modal Scene Fusion-based Hyperspectral Image Synthesis](https://www.sciencedirect.com/science/article/abs/pii/S1566253524001970).

## Usage

### Step 0: Data preparation

Download the Chikusei dataset (HSI) from https://naotoyokoya.com/Download.html, and divide and crop the HSI into several mat files of spatial size $256\times 256\times 59$ for training the unmixing network. Put them in `./dataset/trains/` and `./dataset/evals/`

Download the HSRS-SC dataset (HSI) from http://www.cjig.cn/jig/ch/reader/view_abstract.aspx?file_no=20210805, and resample the HSIs into $256\times 256\times 59$ for validation of the unmixing network. Put them in `./dataset/tests/`.

Download the AID dataset (RGB) from https://hyper.ai/datasets/5446, and resize the images into $256\times 256$. Put them in `./datasets/RGB/ `


### Step 1:   Scene-based Unmixing
For training the unmixing net, change the file path and run the following code.

`python 1_scene-based-unmixing.py train`

After training, run the following code to infer the abundance maps of external RGB datasets. 

`python 1_scene-based-unmixing.py infer` 

After that, we can obtain the inferred abundance of RGB dataset in `./datasets/inferred_abu/`.

### Step 2:  Abundance-based Diffusion
For training the Abundance-based Diffusion, run the following code:

`python 2_abundance-based-diffusion.py -p train`

After training, modify the 'resume_state' in the `./config/*.json` file, and run:

`python 2_abundance-based-diffusion.py -p val`

After that, we can obtain the synthesized abundance in `./experiments/ddpm/\*/mat_results/`.

### Step 3:  Fusion-based generation

Change the `train_path` (path of synthesized abundances) and the `model_name`(the trained model of the unmixing net)

Run the following code to obtain the synthetic HSIs:

`python 3_fusion-based_generation.py`

After that, we can obtain the synthesized HSIs in `./experiments/fusion/HSI/` and its corresponding false-color images in `./experiments/fusion/RGB/`.

## Citation

If you find this work useful, please cite our paper:

```
@article{pan2024abundance,
  title={From the abundance perspective: Multi-modal scene fusion-based hyperspectral image synthesis},
  author={Pan, Erting and Yu, Yang and Mei, Xiaoguang and Huang, Jun and Ma, Jiayi},
  journal={Information Fusion},
  pages={102419},
  year={2024},
  publisher={Elsevier}
}
```

## Contact 

Feel free to open an issue if you have any question. You could also directly contact us through email at [panerting@whu.edu.cn](mailto:panerting@whu.edu.cn) (Erting Pan)






