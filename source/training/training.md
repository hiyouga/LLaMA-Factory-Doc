# 训练

您可以使用命令行或者通过WebUi进行训练推理。

## 命令行

假设您需要使用Llama3-8B-Instruct模型来运行LoRa微调、推理和合并。

### 微调

您可以修改对应文件`examples/train_lora/llama3_lora_sft.yaml`来自行配置数据集、截断长度以及输出路径等参数。修改完毕后运行`llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml`即可。

```yaml
# examples/train_lora/llama3_lora_sft.yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: identity,alpaca_en_demo
template: llama3
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama3-8b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

### 推理

微调完成后，您可以通过`llamafactory-cli chat examples/inference/llama3_lora_sft.yaml`来进行推理。在`examples/inference/llama3_lora_sft.yaml`中，我们指定了适应器路径，如果您在微调时修改了输出路径，请在推理时也根据实际情况修改。

```yaml
# examples/inference/llama3_lora_sft.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft
template: llama3
finetuning_type: lora
```

### 合并

最后，您可以通过`llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml`来合并参数。	

<div style="padding: 10px; margin-bottom: 10px; border: 1px solid #1a73e8; background-color: #f8ffff;">
    如果您需要使用其他微调方法，请查阅`examples`目录下的示例文件并根据自身需要修改参数实现。
</div>