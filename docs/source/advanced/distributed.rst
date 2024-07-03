分布训练
==================
LLaMA-Factory支持单机多卡和多机多卡分布式训练。同时也支持 :ref:`NativeDDP<NativeDDP>`, :ref:`fsdp<fsdp>` 和 :ref:`deepspeed <deepspeed>` 三种分布式训练方式。


单机多卡
------------------------

* :ref:`NativeDDP单机多卡 <torchrun单机多卡>`

* :ref:`fsdp单机多卡 <fsdp单机多卡>`

* :ref:`deepspeed单机多卡 <deepspeed单机多卡>`


多机多卡
-----------------------------
* :ref:`NativeDDP多机多卡 <torchrun多机多卡>`
* :ref:`fsdp多机多卡 <fsdp多机多卡>`
* :ref:`deepspeed多机多卡 <deepspeed多机多卡>`



.. _NativeDDP:

NativeDDP
~~~~~~~~~~~~~~~~~~~~~~~~~

NativeDDP是PyTorch的一种分布式训练方式，您可以通过以下命令启动训练：

.. _torchrun:

.. torchrun
.. ~~~~~~~~~~~~~~~~~~~~~~~~~

.. _torchrun单机多卡:

单机多卡
+++++++++++++++++++


.. code-block:: bash

    FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml

如果 ``CUDA_VISIBLE_DEVICES`` 没有指定，则默认使用所有GPU。


.. _torchrun多机多卡:

多机多卡
++++++++++++++++++++

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
      - 是否强制使用torchrun
    * - NNODES
      - 节点数量
    * - RANK
      - 各个节点的rank。
    * - MASTER_ADDR
      - 主节点的地址。
    * - MASTER_PORT
      - 主节点的端口。



.. _fsdp:

fsdp
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _fsdp单机多卡:

.. _fsdp多机多卡:


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
    * 请确保 ``num_processes`` 和实际使用的总GPU数量一致 


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



.. _deepspeed:


deepspeed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DeepSpeed是由微软开发的一个开源深度学习优化库，旨在提高大模型训练的效率和速度。为了在训练中使用deepspeed，您需要先估计训练任务的显存大小，再根据任务需求与资源情况选择合适的ZeRO阶段。

简单来说：从ZeRO-1到ZeRO-3，阶段数越高，显存需求越小，但是训练速度也依次变慢。此外，设置 ``offload_param=cpu`` 参数会大幅减小显存需求，但会极大地使训练速度减慢。因此，如果您有足够的显存，
应当使用ZeRO-1，并且确保 ``offload_param=none``。

LLaMA-Factory提供了使用不同阶段的deepspeed配置文件的示例。包括：

* :ref:`ZeRO-0` (不开启)
* :ref:`ZeRO-2`
* :ref:`ZeRO-2+offload <zero2O>`
* :ref:`ZeRO-3`
* :ref:`ZeRO-3+offload <zero3O>`

.. note::
    `https://huggingface.co/docs/transformers/deepspeed <https://huggingface.co/docs/transformers/deepspeed/>`_ 提供了更为详细的介绍。



.. _deepspeed单机多卡:

单机多卡
++++++++++++++++++++++

.. code-block:: bash

    llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml


.. _deepspeed多机多卡:

多机多卡
+++++++++++++++++++++

你可以使用 ``deepspeed`` 命令来启动多机多卡训练。

.. code-block:: bash

    deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
    your_program.py <normal cl args> --deepspeed ds_config.json

LLaMA-Factory也支持deepspeed的多机多卡训练，您可以通过以下命令启动：

.. code-block:: bash

    FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml
    FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml



.. _ZeRO-0:

ZeRO-0
*************************

.. code-block:: yaml

    ### ds_z0_config.json
    {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": true,
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 0,
            "allgather_partitions": true,
            "allgather_bucket_size": 5e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": true,
            "round_robin_gradients": true
        }
    }



.. _ZeRO-2:


ZeRO-2
**************************

.. code-block:: yaml

    ### ds_z2_config.json
    {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": true,
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": true,
            "allgather_bucket_size": 5e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": true,
            "round_robin_gradients": true
        }
    }



.. _zero2O:

ZeRO-2+offload
*************************


.. code-block:: yaml

    ### ds_z2_offload_config.json
    {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true,
        "round_robin_gradients": true
    }
    }


.. _ZeRO-3:

ZeRO-3
****************************

.. code-block:: yaml

    ### ds_z3_config.json
    {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
    }


.. _zero3O:

ZeRO-3+offload
*****************************

.. code-block:: yaml

    ### ds_z3_offload_config.json
    {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
        },
        "offload_param": {
        "device": "cpu",
        "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
    }


.. tip:: 

    `https://www.deepspeed.ai/docs/config-json/ <https://www.deepspeed.ai/docs/config-json/>`_ 提供了关于deepspeed配置文件的更详细的介绍。

