# CFP: Collaborative filter pruning for efficient automatic surface defect detection


## Usage

### Code Requirements
The environment requirements for the code in this repository are provided in requirements.txt. They can be installed in bulk with the following command:

```
pip install -r requirements.txt
```

### Dataset


- [X-SDD](https://pan.baidu.com/s/1EZJIWr7rUWI7l9_0BQVeOAw)

   https://pan.baidu.com/s/1EZJIWr7rUWI7l9_0BQVeOA 提取码:fuyq
- [GC10-DET](https://pan.baidu.com/s/1_P2fOt5LV6Y5CA4ZVSMcWQ)

   https://pan.baidu.com/s/1_P2fOt5LV6Y5CA4ZVSMcWQ 提取码:7f0e
- [ELPV-DC](https://pan.baidu.com/s/1iAdlWv1C0sYJ7XEhH7Em8g)

   https://pan.baidu.com/s/1iAdlWv1C0sYJ7XEhH7Em8g 提取码:4282

### 1. Train Baseline


#### RepVGG-A0 on GC10-DET dataset
```shell
python train_baseline_RepVGGA0.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--gpu 6
```

```shell
python train_class_balance_RepVGGA0.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--gpu 6
```

```shell
python RepVGG_convert.py \
--load ./RepVGG_RL/model_best.pth.tar \
--save ./RepVGG_RL/RepVGG_convert.pth.tar \
--num_classes 10 \
```

#### RepVGG-A0 on X-SDD dataset
```shell
python train_baseline_RepVGGA0.py \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--gpu 6
```

```shell
python train_class_balance_RepVGGA0.py \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--gpu 6
```

#### VGG-16 on GC10-DET dataset
```shell
python train_baseline_VGG.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--gpu 6
```

```shell
python train_class_balance_VGG.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--gpu 6
```
#### VGG-16 on X-SDD dataset
```shell
python train_baseline_VGG.py \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--gpu 6
```

```shell
python train_class_balance_VGG.py \
--dataset X-SDD \
--input_size 128 \
--num_classes 7 \
--gpu 6
```

#### DDDN on GC10-DET dataset
```shell
python train_baseline_DDDN.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--gpu 6
```
```shell
python train_class_balance_DDDN.py \
--dataset GC10-DET \
--input_size 224 \
--num_classes 10 \
--gpu 6
```

#### DDDN on X-SDD dataset
```shell
python train_baseline_DDDN.py \
--dataset X-SDD \
--input_size 224 \
--num_classes 10 \
--gpu 6
```
```shell
python train_class_balance_DDDN.py \
--dataset X-SDD \
--input_size 224 \
--num_classes 10 \
--gpu 6
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
--pretrain_dir  ./RepVGG_RL/RepVGG_convert.pth.tar \
--teacher_dir ./RepVGG_CL/RepVGG_convert.pth.tar \
--rank_conv_prefix ./Rank_RepVGG-A0 \
--hierarchical_conv_prefix ./HC_RepVGG-A0 \
--compress_rate [0.]+[0.25]*6+[0.35]*14  \
--gpu 6
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
--compress_rate [0.]+[0.25]*6+[0.35]*14  \
--gpu 6
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
--compress_rate [0.4]*5+[0.5]*7  \
--gpu 6
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
--compress_rate [0.4]*5+[0.5]*7  \
--gpu 6
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
--compress_rate [0.]+[0.2]+[0.2]*2+[0.25]*4+[0.25]*8+[0.30]*8+[0.30]*2 \
--gpu 6
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
--gpu 6 
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
--gpu 6
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
--gpu 6
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
--gpu 6
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
--gpu 6
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
--gpu 6 
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
--gpu 6 
```


### Experimental results

We provide our pretrained models and pruned models. Moreover, we provide the results of matrix rank and FCC.

- [X-SDD](https://pan.baidu.com/s/1uaT0j2FzbsoTYV1WnBbv1w)

   https://pan.baidu.com/s/1uaT0j2FzbsoTYV1WnBbv1w 提取码:1ui5
- [GC10-DET](https://pan.baidu.com/s/1_P2fOt5LV6Y5CA4ZVSMcWQ)

   https://pan.baidu.com/s/1_P2fOt5LV6Y5CA4ZVSMcWQ 提取码:7f0e
- [ELPV-DC](https://pan.baidu.com/s/1iAdlWv1C0sYJ7XEhH7Em8g)

   https://pan.baidu.com/s/1iAdlWv1C0sYJ7XEhH7Em8g 提取码:4282



