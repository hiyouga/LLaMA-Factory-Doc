lora合并
#################
当我们基于预训练模型训练好LoRA适配器后，我们不希望在每次推理的时候分别加载预训练模型和LoRA适配器，因此我们需要将预训练模型和LoRA适配器合并。




量化（Quantization）通过数据精度压缩有效地减少了显存使用并加速推理。LLaMA-Factory支持多种量化方法，包括:

* AQLM
* AWQ
* GPTQ
* QLoRA
* ...

GPTQ等后训练量化方法(Post Training Quantization)是一种在训练后对预训练模型进行量化的方法。
我们通过量化技术将高精度表示的预训练模型转换为低精度的模型，从而在避免过多损失模型性能的情况下减少显存占用并加速推理，
我们希望低精度数据类型在有限的表示范围内尽可能地接近高精度数据类型的表示，因此我们需要校准数据集 ``calibration dataset``

.. note::
    在进行模型合并时，请指定：
    
    * ``model_name_or_path``: 预训练模型的名称或路径
    * ``template``: 模型模板
    * ``export_dir``: 导出路径
    * ``export_quantization_bit``: 量化位数
    * ``export_quantization_dataset``: 量化校准数据集
    * ``export_size``: 最大导出模型文件大小
    * ``export_device``: 导出设备
    * ``export_legacy_format``: 是否使用旧格式导出

.. code-block:: yaml

    ### examples/merge_lora/llama3_gptq.yaml
    ### model
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    template: llama3

    ### export
    export_dir: models/llama3_gptq
    export_quantization_bit: 4
    export_quantization_dataset: data/c4_demo.json
    export_size: 2
    export_device: cpu
    export_legacy_format: false


QLoRA是一种在4-bits量化模型基础上使用LoRA方法进行训练的技术。它在极大地保持了模型性能的同时大幅减少了显存占用和推理时间。

.. warning:: 
    不要使用量化模型或设置量化位数 ``quantization_bit``


.. code-block:: yaml

    ### examples/merge_lora/llama3_q_lora.yaml
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




