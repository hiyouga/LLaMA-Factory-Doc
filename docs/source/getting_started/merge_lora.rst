LoRA 合并
#################

合并
~~~~~~~~~~~~~~~~~~~~~~~

当我们基于预训练模型训练好 LoRA 适配器后，我们不希望在每次推理的时候分别加载预训练模型和 LoRA 适配器，因此我们需要将预训练模型和 LoRA 适配器合并导出成一个模型，并根据需要选择是否量化。根据是否量化以及量化算法的不同，导出的配置文件也有所区别。

您可以通过 ``llamafactory-cli export merge_config.yaml`` 指令来合并模型。其中 ``merge_config.yaml`` 需要您根据不同情况进行配置。

``examples/merge_lora/llama3_lora_sft.yaml`` 提供了合并时的配置示例。

.. code-block:: yaml

    ### examples/merge_lora/llama3_lora_sft.yaml
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


.. note::
    * 模型 ``model_name_or_path`` 需要存在且与 ``template`` 相对应。 ``adapter_name_or_path`` 需要与微调中的适配器输出路径 ``output_dir`` 相对应。
    * 合并 LoRA 适配器时，不要使用量化模型或指定量化位数。您可以使用本地或下载的未量化的预训练模型进行合并。


量化
~~~~~~~~~~~~~~~~~~~~~~~

在完成模型合并并获得完整模型后，为了优化部署效果，人们通常会基于显存占用、使用成本和推理速度等因素，选择通过量化技术对模型进行压缩，从而实现更高效的部署。

量化（Quantization）通过数据精度压缩有效地减少了显存使用并加速推理。LLaMA-Factory 支持多种量化方法，包括:

* AQLM
* AWQ
* GPTQ
* QLoRA
* ...

GPTQ 等后训练量化方法(Post Training Quantization)是一种在训练后对预训练模型进行量化的方法。我们通过量化技术将高精度表示的预训练模型转换为低精度的模型，从而在避免过多损失模型性能的情况下减少显存占用并加速推理，我们希望低精度数据类型在有限的表示范围内尽可能地接近高精度数据类型的表示，因此我们需要指定量化位数 ``export_quantization_bit`` 以及校准数据集 ``export_quantization_dataset``。

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

下面提供一个配置文件示例：

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


QLoRA 是一种在 4-bit 量化模型基础上使用 LoRA 方法进行训练的技术。它在极大地保持了模型性能的同时大幅减少了显存占用和推理时间。

.. warning:: 
    不要使用量化模型或设置量化位数 ``quantization_bit``

下面提供一个配置文件示例：

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

