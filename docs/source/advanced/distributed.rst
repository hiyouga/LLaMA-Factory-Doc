分布训练
==================

如果您需要使用多机或者多个显卡进行训练，下面提供了一个例子以供参考。

.. code-block:: bash

    FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
    
    FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 \
    llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml


.. list-table::
    :widths: 30 70  
    :header-rows: 1

    * - 变量名
      - 介绍
    * - FORCE_TORCHRUN
      - 是否强制使用torch.distributed.launch。
    * - NNODES
      - 显卡数量。  
    * - RANK
      - 各个节点的rank。
    * - MASTER_ADDR
      - 主节点的地址。
    * - MASTER_PORT
      - 主节点的端口。



:ref:`fsdp <fsdp>` 、 


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
