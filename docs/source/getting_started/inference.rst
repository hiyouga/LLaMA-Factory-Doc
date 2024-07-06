推理
==========================


以下指令提供了使用 ``llamafactory-cli`` 进行推理的示例。您需要根据自身需求自行配置。

.. code-block:: bash

    llamafactory-cli chat inference_config.yaml

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


多模态模型
----------------------

对于多模态模型，您可以运行以下指令进行推理。

.. code-block:: bash

    llamafactory-cli webchat llava.yaml

``llava.yaml`` 的配置如下：

.. code-block:: yaml

    model_name_or_path: llava-hf/llava-1.5-7b-hf
    template: vicuna
    visual_inputs: true




vllm推理框架
------------------------
若使用vllm推理框架，请在配置中指定： ``infer_backend`` 与 ``vllm_enforce_eager``。

.. code-block:: yaml

    ### examples/inference/llama3_vllm.yaml
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    template: llama3
    infer_backend: vllm
    vllm_enforce_eager: true




推理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``examples/inference/llama3_lora_sft.yaml`` 提供了推理时的配置示例。

.. code-block:: yaml

    ### examples/inference/llama3_lora_sft.yaml
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    adapter_name_or_path: saves/llama3-8b/lora/sft
    template: llama3
    finetuning_type: lora

.. note::
    模型 ``model_name_or_path`` 需要存在且与 ``template`` 相对应。
