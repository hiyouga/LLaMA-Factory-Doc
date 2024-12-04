推理
==========================

LLaMA-Factory 支持多种推理方式。

您可以使用 ``llamafactory-cli chat inference_config.yaml`` 或 ``llamafactory-cli webchat inference_config.yaml`` 进行推理与模型对话。对话时配置文件只需指定原始模型 ``model_name_or_path`` 和 ``template`` ，并根据是否是微调模型指定 ``adapter_name_or_path`` 和 ``finetuning_type``。

如果您希望向模型输入大量数据集并记录推理输出，您可以使用 ``llamafactory-cli train inference_config.yaml`` 使用数据集或 ``llamafactory-cli api`` 使用 api 进行批量推理。

.. note::
    使用任何方式推理时，模型 ``model_name_or_path`` 需要存在且与 ``template`` 相对应。

原始模型推理配置
----------------------------
对于原始模型推理， ``inference_config.yaml`` 中 只需指定原始模型 ``model_name_or_path`` 和 ``template`` 即可。

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

    llamafactory-cli webchat examples/inferece/llava1_5.yaml

``examples/inference/llava1_5.yaml`` 的配置示例如下：

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



.. _批量推理:

批量推理
-------------------------

数据集
~~~~~~~~~~~~~~~~~~~~~~~
使用数据集批量推理时，您需要指定模型、适配器（可选）、评估数据集、输出路径等信息并且指定 ``do_predict`` 为 ``true``。
下面提供一个 **示例**,您可以通过 ``llamafactory-cli train examples/train_lora/llama3_lora_predict.yaml`` 使用数据集进行批量推理。

如果您需要多卡推理，则需要在配置文件中指定 ``deepspeed`` 参数。

.. code-block:: yaml

    # examples/train_lora/llama3_lora_predict.yaml
    ### model
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    adapter_name_or_path: saves/llama3-8b/lora/sft
    
    deepspeed: examples/deepspeed/ds_z3_config.yaml # deepspeed配置文件
    
    ### method
    stage: sft
    do_predict: true
    finetuning_type: lora

    ### dataset
    eval_dataset: identity,alpaca_en_demo
    template: llama3
    cutoff_len: 1024
    max_samples: 50
    overwrite_cache: true
    preprocessing_num_workers: 16

    ### output
    output_dir: saves/llama3-8b/lora/predict
    overwrite_output_dir: true

    ### eval
    per_device_eval_batch_size: 1
    predict_with_generate: true
    ddp_timeout: 180000000

.. note::

    只有 ``stage`` 为 ``sft`` 的时候才可设置 ``predict_with_generate`` 为 ``true``


api
~~~~~~~~~~~~~~~~~
如果您需要使用 api 进行批量推理，您只需指定模型、适配器（可选）、模板、微调方式等信息。

下面是一个配置文件的示例：

.. code-block:: yaml

    # examples/inference/llama3_lora_sft.yaml
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    adapter_name_or_path: saves/llama3-8b/lora/sft
    template: llama3
    finetuning_type: lora


下面是一个启动并调用 api 服务的示例：

您可以使用 ``API_PORT=8000 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api examples/inference/llama3_lora_sft.yaml`` 启动 api 服务并运行以下示例程序进行调用：

.. code-block:: python

    # api_call_example.py
    from openai import OpenAI
    client = OpenAI(api_key="0",base_url="http://0.0.0.0:8000/v1")
    messages = [{"role": "user", "content": "Who are you?"}]
    result = client.chat.completions.create(messages=messages, model="meta-llama/Meta-Llama-3-8B-Instruct")
    print(result.choices[0].message)




