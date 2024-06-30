# 训练

您可以使用命令行或者通过WebUi进行训练推理。

## 命令行

以下是使用命令行分别进行微调、推理和合并的示例。该例子提供了一个使用Llama3-8B-Instruct模型运行LoRA微调、推理和合并的流程。

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### 微调

`examples/train_lora/llama3_lora_sft.yaml`提供了微调时的配置示例，如果您需要使用其他模型或其他数据集，则需要根据需求自行配置。

```yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct # 模型名称或路径

### method
stage: sft # 训练阶段，可选:rm(reward modeling),pt(pretrain),sft(Supervised Fine-Tuning),PPO,DPO,KTO,ORPO
do_train: true # true用于训练,false用于评估
finetuning_type: lora #微调方式 
lora_target: all # 采取Lora方法的目标模块

### dataset
dataset: identity,alpaca_en_demo # 使用的数据集，使用","分隔多个数据集
template: llama3 # 数据集模板
cutoff_len: 1024 
max_samples: 1000 
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/sft # 输出路径
logging_steps: 10 # 日志输出步数间隔
save_steps: 500 # 模型保存间隔
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1 # 每个设备上训练的批次大小,数字越大占用显存越大
gradient_accumulation_steps: 8 # 梯度积累步数，数字越大占用显存越小
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine # 学习率曲线
warmup_ratio: 0.1 #预热学习率，初始学习率lr = lr * warmup_ratio
bf16: true # 是否使用bf16格式
ddp_timeout: 180000000 # 分布式数据并行最大等待时间

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps 
eval_steps: 500

```

<div style="padding: 1px; margin-bottom: 1px; border: 1px solid #1a73e8; background-color: #f8ffff;">
    <strong>注意:</strong> 模型model_name_or_path、数据集dateset需要存在且与template相对应。</li>
</div>



### 推理

`examples/inference/llama3_lora_sft.yaml`提供了推理时的配置示例。

```yaml
# examples/inference/llama3_lora_sft.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft
template: llama3
finetuning_type: lora
```

<div style="padding: 1px; margin-bottom: 1px; border: 1px solid #1a73e8; background-color: #f8ffff;">
    <strong>注意:</strong> 模型model_name_or_path需要存在且与template相对应。</li>
</div>

### 合并

`examples/merge_lora/llama3_lora_sft.yaml`提供了合并时的配置示例。

```yaml
# examples/merge_lora/llama3_lora_sft.yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft
template: llama3
finetuning_type: lora

### export
export_dir: models/llama3_lora_sft
export_size: 2
export_device: cpu
export_legacy_format: false
```

<div style="padding: 5px; margin-bottom: 5px; border: 1px solid #1a73e8; background-color: #f8ffff;">
    <strong>注意:</strong> 
    <li>模型model_name_or_path需要存在且与template相对应</li><li>adapter_name_or_path需要与微调中的适配器输出路径output_dir相对应。</li></pre>
<li>合并LoRA适配器时，不要使用量化模型或量化位数</li>
</div>



## 关于训练参数

LLaMA-Factory支持多种训练策略、训练精度及算法，下面提供了更多关于训练参数的介绍。

### 参数介绍	

#### Freeze

需要冻结模型时，请将`finetuning_type`设置为`freeze`并且设置`FreezeArguments`中的参数：

| 参数名称                     | 类型 | 介绍                                                         |
| ---------------------------- | ---- | ------------------------------------------------------------ |
| freeze_trainable_layers      | int  | 可训练层的数量。正数表示最后 n 层被设置为可训练的，负数表示前 n 层被设置为可训练的。默认值为`2` |
| freeze_trainable_modules     | str  | 可训练层的名称。使用`all`来指定所有模块。默认值为`all`       |
| freeze_extra_modules[非必须] | str, | 除了隐藏层外可以被训练的模块名称，被指定的模块将会被设置为可训练的。使用逗号分隔多个模块。默认值为`None` |

#### LoRA（Low-Rank Adaptation）

需要进行LoRA训练时，请设置`LoraArguments`中的参数。

| 参数名称                      | 类型   | 介绍                                                         |
| ----------------------------- | ------ | ------------------------------------------------------------ |
| additional_target[非必须]     | [str,] | 除LoRA层之外设置为可训练并保存在最终检查点中的模块名称。使用逗号分隔多个模块。默认值为`None` |
| lora_alpha[非必须]            | int    | LoRA 缩放系数。一般情况下为lora_rank * 2,默认值为`None`      |
| lora_dropout                  | float  | LoRA微调中的dropout率。默认值为`0`                           |
| lora_rank                     | int    | LoRA微调的本征维数$r$，$r$越大可训练的参数越多。默认值为`8`  |
| lora_target                   | str    | 应用LoRA方法的模块名称。使用逗号分隔多个模块，使用`all`指定所有模块。默认值为`all` |
| loraplus_lr_ratio[非必须]     | float  | LoRA+学习率比例($r = \frac{\eta_A}{\eta_B}$)。$\eta_A,\eta_B$分别是adapter matrices A与B的学习率。实验表明，将这个值设置为`16`会取得较好的初始结果。当任务较为复杂时需要将这个值设置得大一些。默认值为`None` |
| loraplus_lr_embedding[非必须] | float  | LoRA+嵌入层的学习率,默认值为`1e-6`                           |
| use_rslora                    | bool   | 是否使用秩稳定LoRA(Rank-Stabilized LoRA)，默认值为`False`    |
| use_dora                      | bool   | 是否使用权重分解LoRA（Weight-Decomposed LoRA），默认值为`False` |
| pissa_init                    | bool   | 是否初始化PiSSA适配器，默认值为`False`                       |
| pissa_iter                    | int    | PiSSA中FSVD执行的迭代步数。使用`-1`将其禁用，默认值为`16`    |
| pissa_convert                 | bool   | 是否将PiSSA适配器转换为正常的LoRA适配器，默认值为`False`     |
| create_new_adapter            | bool   | 是否创建一个具有随机初始化权重的新适配器，默认值为`False`    |

#### Finetuning

当您需要进行模型微调时，可以配置`FinetuningArguments`类中的参数。

| 参数名称            | 类型    | 介绍                                                         |
| ------------------- | ------- | ------------------------------------------------------------ |
| pure_bf16           | bool    | 是否在纯bf16精度下训练模型（不使用AMP），默认值为`False`     |
| stage               | Literal | 训练的阶段，可选值有：`pt`（pre-training）、`sft`（supervised fine-tuning）、`rm`（reward modeling）、`ppo`（Proximal Policy Optimization）、`dpo`（Deep Preference Optimization）、`kto`（Keyframe Threshold Optimization）。默认值为`sft` |
| finetuning_type     | Literal | 使用的微调方法，可选值有：`lora`、`freeze`、`full`，默认值为`lora` |
| use_llama_pro       | bool    | 是否只令扩展块中的参数可训练，默认值为`False`。              |
| freeze_vision_tower | bool    | 是否在MLLM训练中冻结vision tower，默认值为`True`。           |
| train_mm_proj_only  | bool    | 是否仅训练MLLM中的多模态投影器，默认值为`False`。            |
| plot_loss           | bool    | 是否保存训练损失曲线，默认值为`False`。                      |

