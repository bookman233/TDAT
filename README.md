# Taxonomy Driven Fast Adversarial Training

## Environments

- python 3.7.13
- torch 1.13.1
- torchvision 0.13

## Files

### TDAT.py

- Train ResNet18 on CIFAR-10 with our proposed TDAT.

`python TDAT.py --batch-m 0.75 --delta-init "random" --out-dir "TDAT" --log "CIFAR10.log" --model "ResNet18" --lamda 0.6 --inner-gamma 0.15 --outer-gamma 0.15 --save-epoch 1`

### ResNet18_CIFAR10_TDAT_robustAcc_0.5685_clean_acc_0.8225.pt

- Checkpoint on CIFAR-10 with our method.
- We had to remove this checkpoint from the supplementary materials due to its 50MB size limit. Executing the TDAT.py file with the above command can obtain the best results as well.

### models

- This folder holds the codes for backbones.

### CIFAR10/CIFAR100/TinyImageNet/ImageNet100

- These folders store the corresponding training log.

## Trained Models
[TDAT with ResNet-18 on CIFAR-10 dataset](https://drive.google.com/file/d/1fPYwjz2V9wibfdWlopip0tfK4IB0KS9o/view?usp=drive_link)
