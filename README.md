# CFP: Collaborative filter pruning for efficient automatic surface defect detection

## Pretrained models
RepVGG-A0  [Baidu Yun](https://pan.baidu.com/s/1TQGk694k-dY2HvNrv7--7g?pwd=9auu) [Google Yun](https://drive.google.com/file/d/17hhNL-pE4yp0wlCsHpBmsDYtPOt3OL6q/view?usp=drive_link) VGG-16 [Baidu Yun](https://pan.baidu.com/s/1MF_hrfixW8ZMCp5Yaq_vXA?pwd=wm34) [Google Yun](https://drive.google.com/file/d/1ffb9XvHcT4YwHRepmBctBo0IQzXZO2Io/view?usp=drive_link)


## Usage


### 1. Train Baseline


#### RepVGG-A0 on GC10-DET dataset
```shell
python train_baseline_RepVGGA0.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--pretrain_dir ./RepVGG-A0-train.pth \
--job_dir ./Original_RepVGGA0 \
--gpu 2
```

```shell
python train_class_balance_RepVGGA0.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--pretrain_dir ./RepVGG-A0-train.pth \
--job_dir ./Original_RepVGGA0 \
--gpu 2
```

```shell
python RepVGG_convert.py \
--load ./Original_RepVGGA0/model_best.pth.tar \
--save ./Original_RepVGGA0/RepVGG_convert.pth.tar \
--num_classes 10 
```



#### RepVGG-A0 on X-SDD dataset
```shell
python train_baseline_RepVGGA0.py \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--pretrain_dir ./RepVGG-A0-train.pth \
--job_dir ./Original_RepVGGA0 \
--gpu 2
```

```shell
python train_class_balance_RepVGGA0.py \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--pretrain_dir ./RepVGG-A0-train.pth \
--job_dir ./Class_balance_RepVGGA0 \
--gpu 2
```

```shell
python RepVGG_convert.py \
--load ./Class_balance_RepVGGA0/model_best.pth.tar \
--save ./Class_balance_RepVGGA0/RepVGG_convert.pth.tar \
--num_classes 7 
```

#### VGG-16 on GC10-DET dataset
```shell
python train_baseline_VGG.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--job_dir ./Original_VGG \
--gpu 2
```

```shell
python train_class_balance_VGG.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--job_dir ./Class_balance_VGG \
--gpu 2
```
#### VGG-16 on X-SDD dataset
```shell
python train_baseline_VGG.py \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--job_dir ./Original_VGG \
--gpu 2
```

```shell
python train_class_balance_VGG.py \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--job_dir ./Class_balance_VGG \
--gpu 2
```

#### DDDN on GC10-DET dataset
```shell
python train_baseline_DDDN.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--gpu 2
```

```shell
python train_class_balance_DDDN.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--gpu 2
```

#### DDDN on X-SDD dataset
```shell
python train_baseline_DDDN.py \
--dataset X-SDD \
--input_size 224 \
--num_classes 7 \
--job_dir ./Original_DDDN \
--gpu 2
```
```shell
python train_class_balance_DDDN.py \
--dataset X-SDD \
--input_size 224 \
--num_classes 7 \
--job_dir ./Class_balance_DDDN \
--gpu 2
```

### 2. Generate Feature Maps.
```shell
python calculate_feature_maps.py \
--arch vgg_16_bn \
--dataset X-SDD \
--batch_size 10 \
--input_size 128 \
--num_classes 7 \
--repeat 1 \
--pretrain_dir ./VGG_RL/model_best.pth.tar \
--gpu 5
```

### 3. Calculate Distance Matrix.
```shell
python calculate_distance_matrix.py \
--arch DDDN \
--num_layers 26 \
--gpu 1 
```

### 4. Calculate Matrix Rank.
```shell
python calculate_matrix_rank.py \
--arch DDDN \
--num_layers 26 \
--gpu 2
```
### 5. Perform the FCC.
```shell
python calculate_hierarchical_clustering.py \
--arch vgg_16_bn \
--num_layers 12 \
--scale 2 \
--distance_dir ./Distance_vgg_16_bn \
--gpu 3
```

### 6. Prune and Fine-tune CNN Models.
#### RepVGG-A0 on X-SDD dataset

```shell
python train_pruning_CFP.py \
--arch RepVGGA0 \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--job_dir ./pruned_CFP_RepVGGA0 \
--pretrain_dir  ./Original_RepVGGA0/RepVGG_convert.pth.tar \
--teacher_dir ./Class_balance_RepVGGA0/RepVGG_convert.pth.tar \
--rank_conv_prefix ./Rank_RepVGG-A0 \
--hierarchical_conv_prefix ./HC_RepVGG-A0 \
--compress_rate [0.]+[0.25]*6+[0.35]*14  \
--gpu 2
```

#### RepVGG-A0 on GC10-DET dataset
```shell
python train_pruning_CFP.py \
--arch RepVGGA0 \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--job_dir ./pruned_CFP_RepVGGA0 \
--pretrain_dir  ./Original_RepVGG/RepVGG_convert.pth.tar \
--teacher_dir ./Original_RepVGG/RepVGG_convert.pth.tar \
--rank_conv_prefix ./Rank_RepVGG-A0 \
--hierarchical_conv_prefix ./HC_RepVGG-A0 \
--compress_rate [0.]+[0.25]*6+[0.35]*14  \
--gpu 2
```

#### VGG-16 on X-SDD dataset
```shell
python train_pruning_CFP.py \
--arch vgg_16_bn \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--job_dir ./pruned_CFP_vgg_16_bn \
--pretrain_dir  ./Original_VGG/model_best.pth.tar \
--teacher_dir ./Class_balance_VGG/model_best.pth.tar \
--rank_conv_prefix ./Rank_vgg_16_bn \
--hierarchical_conv_prefix ./HC_vgg_16_bn \
--compress_rate [0.4]*5+[0.5]*7  \
--gpu 2
```

#### VGG-16 on GC10-DET dataset
```shell
python train_pruning_CFP.py \
--arch vgg_16_bn \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--job_dir ./pruned_CFP_vgg_16_bn \
--pretrain_dir  ./Original_VGG/model_best.pth.tar \
--teacher_dir ./Class_balance_VGG/model_best.pth.tar \
--rank_conv_prefix ./Rank_vgg_16_bn \
--hierarchical_conv_prefix ./HC_vgg_16_bn \
--compress_rate [0.4]*5+[0.5]*7  \
--gpu 2
```


#### DDDN on X-SDD dataset

```shell
python train_pruning_CFP.py \
--arch DDDN \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--job_dir ./pruned_CFP_DDDN \
--pretrain_dir ./Original_DDDN/model_best.pth.tar \
--teacher_dir ./Class_balance_DDDN/model_best.pth.tar \
--rank_conv_prefix ./Rank_DDDN \
--hierarchical_conv_prefix ./HC_DDDN \
--compress_rate [0.]+[0.2]+[0.2]*2+[0.25]*4+[0.25]*8+[0.30]*8+[0.30]*2 \
--gpu 2 
```

#### DDDN on GC10-DET dataset

```shell
python train_pruning_CFP.py \
--arch DDDN \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--job_dir ./pruned_CFP_DDDN \
--pretrain_dir ./DDDN_RL/model_best.pth.tar \
--teacher_dir ./DDDN_CL/model_best.pth.tar \
--rank_conv_prefix ./Rank_DDDN \
--hierarchical_conv_prefix ./HC_DDDN \
--compress_rate [0.]+[0.2]+[0.2]*2+[0.2]*4+[0.25]*8+[0.25]*8+[0.25]*2 \
--gpu 2 
```

### 7. Evaluate Pruned Model

#### RepVGG-A0 on X-SDD dataset
```shell
python train_pruning_CFP.py \
--arch RepVGGA0 \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--job_dir ./pruned_CFP_RepVGGA0 \
--pretrain_dir  ./RepVGG_RL/RepVGG_convert.pth.tar \
--teacher_dir ./RepVGG_CL/RepVGG_convert.pth.tar \
--rank_conv_prefix ./Rank_RepVGG-A0 \
--hierarchical_conv_prefix ./HC_RepVGG-A0 \
--test_only True \
--test_model_dir ./experimental_results/RepVGGA0/model_best.pth.tar \
--compress_rate [0.]+[0.25]*6+[0.35]*14  \
--gpu 2
```

#### RepVGG-A0 on GC10-DET dataset
```shell
python train_pruning_CFP.py \
--arch RepVGGA0 \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--job_dir ./pruned_CFP_RepVGGA0 \
--pretrain_dir  ./RepVGG_RL/RepVGG_convert.pth.tar \
--teacher_dir ./RepVGG_CL/RepVGG_convert.pth.tar \
--rank_conv_prefix ./Rank_RepVGG-A0 \
--hierarchical_conv_prefix ./HC_RepVGG-A0 \
--test_only True \
--test_model_dir ./experimental_result/RepVGGA0/model_best.pth.tar \
--compress_rate [0.]+[0.25]*6+[0.35]*14  \
--gpu 2
```

#### VGG-16 on X-SDD dataset
```shell
python train_pruning_CFP.py \
--arch vgg_16_bn \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--job_dir ./pruned_CFP_vgg_16_bn \
--pretrain_dir  ./VGG_RL/model_best.pth.tar \
--teacher_dir ./VGG_CL/model_best.pth.tar \
--rank_conv_prefix ./Rank_vgg_16_bn \
--hierarchical_conv_prefix ./HC_vgg_16_bn \
--test_only True \
--test_model_dir ./experimental_results/VGG16/model_best.pth.tar \
--compress_rate [0.4]*5+[0.5]*7  \
--gpu 2
```

#### VGG-16 on GC10-DET dataset
```shell
python train_pruning_CFP.py \
--arch vgg_16_bn \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--job_dir ./pruned_CFP_vgg_16_bn \
--pretrain_dir  ./VGG_RL/model_best.pth.tar \
--teacher_dir ./VGG_CL/model_best.pth.tar \
--rank_conv_prefix ./Rank_vgg_16_bn \
--hierarchical_conv_prefix ./HC_vgg_16_bn \
--test_only True \
--test_model_dir ./experimental_result/VGG16/model_best.pth.tar \
--compress_rate [0.4]*5+[0.5]*7  \
--gpu 2
```

#### DDDN on X-SDD dataset

```shell
python train_pruning_CFP.py \
--arch DDDN \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--job_dir ./pruned_CFP_DDDN \
--pretrain_dir ./DDDN_RL/model_best.pth.tar \
--teacher_dir ./DDDN_CL/model_best.pth.tar \
--rank_conv_prefix ./Rank_DDDN \
--hierarchical_conv_prefix ./HC_DDDN \
--test_only True \
--test_model_dir ./experimental_results/DDDN/model_best.pth.tar \
--compress_rate  [0.]+[0.2]+[0.2]*2+[0.25]*4+[0.25]*8+[0.30]*8+[0.30]*2 \
--gpu 3 
```


#### DDDN on GC10-DET dataset

```shell
python train_pruning_CFP.py \
--arch DDDN \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--job_dir ./pruned_CFP_DDDN \
--pretrain_dir ./DDDN_RL/model_best.pth.tar \
--teacher_dir ./DDDN_CL/model_best.pth.tar \
--rank_conv_prefix ./Rank_DDDN \
--hierarchical_conv_prefix ./HC_DDDN \
--test_only True \
--test_model_dir ./experimental_result/DDDN/model_best.pth.tar \
--compress_rate [0.]+[0.2]+[0.2]*2+[0.2]*4+[0.25]*8+[0.25]*8+[0.25]*2 \
--gpu 3 
```


### Experimental results

We provide our pretrained models and pruned models. Moreover, we provide the results of matrix rank and FCC.

- [Pre-trained Models](https://drive.google.com/drive/folders/1b--dZlvKUUu0rXqMYAtIr0ynHQHuEWDI?usp=sharing)
   - GC10-DET: VGG-16, RepVGG-A0, DDDN.
   - X-SDD: VGG-16, RepVGG-A0, DDDN.
   - ELPV-DC: VGG-16, RepVGG-A0, DDDN.
   - NEu-CLS: VGG-16, ResNet-50, MobileNetV2
  
- [Pruned Models](https://drive.google.com/drive/folders/1b--dZlvKUUu0rXqMYAtIr0ynHQHuEWDI?usp=sharing)
   - GC10-DET: VGG-16, RepVGG-A0, DDDN.
   - X-SDD: VGG-16, RepVGG-A0, DDDN.
   - ELPV-DC: VGG-16, RepVGG-A0, DDDN.
   - NEu-CLS: VGG-16, ResNet-50, MobileNetV2
- [FCC results](https://drive.google.com/drive/folders/1b--dZlvKUUu0rXqMYAtIr0ynHQHuEWDI?usp=sharing)

- [Matrix rank](https://drive.google.com/drive/folders/1b--dZlvKUUu0rXqMYAtIr0ynHQHuEWDI?usp=sharing)


