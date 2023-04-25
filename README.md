# Recursions Are All You Need
This repository holds the official code used in the paper:

Recursions Are All You Need: Towards Efficient Deep Unfolding Networks

The code was run on a Linux-based system (Ubuntu 22.04) on a single Nvidia RTX 3090 GPU, and was written using PyTorch.

## Abstract
The use of deep unfolding networks in compressive sensing (CS) has seen wide success as they provide both simplicity and interpretability. However, since most deep unfolding networks are iterative, this incurs significant redundancies in the network. In this work, we propose a novel recursion-based framework to enhance the efficiency of deep unfolding models. First, recursions are used to effectively eliminate the redundancies in deep unfolding networks. Secondly, we randomize the number of recursions during training to decrease the overall training time. Finally, to effectively utilize the power of recursions, we introduce a learnable unit to modulate the features of the model based on both the total number of iterations and the current iteration index. To evaluate the proposed framework, we apply it to both ISTA-Net+ and COAST. Extensive testing shows that our proposed framework allows the network to cut down as much as 75% of its learnable parameters while mostly maintaining its performance, and at the same time, it cuts around 21% and 42% from the training time for ISTA-Net+ and COAST respectively. Moreover, when presented with a limited training dataset, the recursive models match or even outperform their respective non-recursive baseline.

![Recursive_Framework](/Figures/Recursive_Framework.png)
Figure 1: General architecture of the recursive framework. Compared to general deep unfolding models such as COAST and ISTA-Net+, $R_i$ recursions are used in each recovery block $i$ in the recovery subnet.

## General Instructions
The code is split into two directories. One for COAST and the other one is for ISTA-Net+. Currently, only the recursive COAST codes are readily available, so the recursive ISTA-Net+ code will be uploaded later.

## Training Setup
Download the training data from [here](https://drive.google.com/file/d/14CKidNsC795vPfxFDXa1FH9QuNJKE3cp/view?usp=sharing) to the `data` directory and then run `COAST/TRAIN_COAST.py`.

Arguements:

| *Arguements* | Description | Default Value |
| ------------ | ----------- | ------------- |
| `--start_epoch` | Starting epoch number | 0 |
| `--end_epoch` | Final epoch number | 400 for COAST and 200 for ISTA-Net+ |
| `--RFMU` | Adds the RFMU unit | True |
| `--layer_num` | Number of recovery blocks | 5 and 3 for COAST and ISTA-Net+ respectively |
| `--IPL` | Number of iterations per layer | 4 for COAST and 3 for ISTA-Net+ |
| `--learning_rate` | Sets the learning rate of the Adam optimizer | `1e-4` |
| `--gpu_list` | Selects the GPUs to be used during training (not tested for more than one GPU) | `'0'` |
| `--num_workers` | Number of workers used in the data loader | 10 |
| `--matrix_dir` | Path to the sampling matrices | 'sampling_matrix' |
| `--model_dir` | Path to the trained model (not working currently) | N/A |
| `--data_dir` | Path to the directory holding the data (whether it is for training or validation) | 'data' |
| `--validation_name` | Validation Dataset (can be Set11, BSD68, BSD100, or Urban100) | `'Set11'` |
| `--save_cycle` | Save cycle period to save the model weights (models acheving the best PSNR or SSIM scores are always saved immediately regardless of the save cycle) | 10 |

## Testing
Run `COAST/TEST_COAST.py` and it will print out the results of all the configurations of COAST used in the paper (it is recommended to run them cell by cell).

## Results
![Tables](/Figures/Tables.png)

## Acknowledgement
- Author(s) would like to acknowledge the support received from Saudi Data and AI Authority (SDAIA) and King Fahd University of Petroleum and Minerals (KFUPM) under SDAIA-KFUPM Joint Research Center for Artificial Intelligence.
- In addition, we would like to thank the authors of the papers [ISTA-Net](https://github.com/jianzhangcs/ISTA-Net-PyTorch) and [COAST](https://github.com/jianzhangcs/COAST) for open-sourcing their code. This was very helpful in our work and our code borrows heavily from them.
