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

# 根据需要自行配置
### train 
# ...
### eval
# ... 
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

