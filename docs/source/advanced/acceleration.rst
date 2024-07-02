加速
=====================

LLaMA-Factory支持多种加速技术，包括： :ref:`fsdp <fsdp>` 、 :ref:`flash-attention <flashattn>` 、 :ref:`unsloth <sloth>`  。


.. _fsdp:

fsdp
---------------------------
PyTorch的全切片数据并行技术（Fully Sharded Data Parallel）能让我们处理更多更大的模型。Huggingface提供了便捷的配置功能。
只需运行：

.. code-block:: bash

    accelerate config


根据提示回答一系列问题后，我们就可以生成fsdp所需的配置文件。

当然您也可以根据需求自行配置 ``fsdp_config.yaml`` 。

.. code-block:: yaml

    ### /examples/accelerate/fsdp_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: false
    distributed_type: FSDP
    downcast_bf16: 'no'
    fsdp_config:
        fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
        fsdp_backward_prefetch: BACKWARD_PRE
        fsdp_forward_prefetch: false
        fsdp_cpu_ram_efficient_loading: true
        fsdp_offload_params: true # offload may affect training speed
        fsdp_sharding_strategy: FULL_SHARD
        fsdp_state_dict_type: FULL_STATE_DICT
        fsdp_sync_module_states: true
        fsdp_use_orig_params: true
    machine_rank: 0
    main_training_function: main
    mixed_precision: fp16 # or bf16
    num_machines: 1 # the number of nodes
    num_processes: 2 # the number of GPUs in all nodes
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false

.. note:: 
    * 请确保 ``num_processes`` 和实际使用的GPU数量一致 


随后，我们可以使用以下命令启动训练：

.. code-block:: bash

    accelerate launch \
    --config_file fsdp_config.yaml \
    train.py llm_config.yaml

以下是一个示例，您可以通过在LLaMA-Factory根目录下运行进行尝试。

.. code-block:: bash

    accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/extras/fsdp_qlora/llama3_lora_sft.yaml


.. warning:: 

    不要在 FSDP+QLoRA 中使用 GPTQ/AWQ 模型


.. _flashattn:


flash-attention
----------------------------

`flash-attention  <https://github.com/Dao-AILab/flash-attention/>`_ 能够加快注意力机制的运算速度，同时减少对内存的使用。

如果您想使用flash-attention,请在启动训练时在训练配置文件中添加以下参数：

.. code-block:: yaml 

    flash_attn: fa2



.. _sloth:

unsloth
---------------------------

`unsloth <https://github.com/unslothai/unsloth/>`_ 框架支持 Llama, Mistral, Phi-3, Gemma, Yi, DeepSeek, Qwen等大语言模型并且支持4bit和16bit的QLoRA/LoRA微调，该框架在提高运算速度的同时还减少了显存占用。

如果您想使用unsloth,请在启动训练时在训练配置文件中添加以下参数：

.. code-block:: yaml

    use_unsloth: True 


