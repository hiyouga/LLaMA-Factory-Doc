推理
==========================

您可以在 ``myconfig.yaml`` 中配置好模型名称、模板等参数后使用以下命令进行推理。


.. code-block:: bash

    llamafactory-cli train myconfig.yaml

原始模型推理配置
----------------------------
对于原始模型推理，只需指定原始模型 ``model_name_or_path`` 和 ``template`` 即可。

.. code-block:: yaml

    ### examples/inference/llama3.yaml
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    template: llama3

微调模型推理配置
------------------------------
对于微调模型推理，除原始模型和模板外，还需要指定适配器路径 ``adapter_name_or_path`` 和微调类型 ``finetuning_type``。

.. code-block:: yaml

    ### examples/inference/llama3_lora_sft.yaml
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    adapter_name_or_path: saves/llama3-8b/lora/sft
    template: llama3
    finetuning_type: lora






