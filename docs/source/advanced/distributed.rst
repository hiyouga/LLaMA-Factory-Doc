分布训练
==================
LLaMA-Factory支持单机多卡和多机多卡分布式训练。同时也支持 :ref:`NativeDDP<NativeDDP>`, :ref:`fsdp<fsdp>` 和 :ref:`deepspeed <deepspeed>` 三种分布式引擎。


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

llamafactory-cli
***************************

您可以使用llamafactory-cli启动NativeDDP引擎。

.. code-block:: bash

    FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml

如果 ``CUDA_VISIBLE_DEVICES`` 没有指定，则默认使用所有GPU。


torchrun
*******************************
您也可以使用torchrun指令启动NativeDDP引擎进行单机多卡训练。

.. code-block:: bash

    torchrun  --standalone --nnodes=1 --nproc-per-node=8 train.py 



accelerate
***************************
您还可以使用accelerate启动进行单机多卡训练。

首先运行以下命令，根据需求回答一系列问题后生成配置文件：

.. code-block:: bash

    accelerate config

下面提供一个示例配置文件：

.. code-block:: yaml

    # accelerate_singleNode_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: true
    distributed_type: MULTI_GPU
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    gpu_ids: all
    machine_rank: 0
    main_training_function: main
    mixed_precision: fp16
    num_machines: 1
    num_processes: 8
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false


您可以通过运行以下指令开始训练:

.. code-block:: bash

    accelerate launch \
    --config_file accelerate_singleNode_config.yaml \
    train.py llm_config.yaml

.. _torchrun多机多卡:

多机多卡
++++++++++++++++++++

llamafactory-cli
*******************

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

torchrun
******************************

您也可以使用 ``torchrun`` 指令启动NativeDDP引擎进行多机多卡训练。

.. code-block:: bash
    
    torchrun --master_port 29500 --nproc_per_node=8 --nnodes=2 --node_rank=0  \
    --master_addr=192.168.0.1  train.py
    torchrun --master_port 29500 --nproc_per_node=8 --nnodes=2 --node_rank=1  \
    --master_addr=192.168.0.1  train.py

accelerate
***************************
您还可以使用accelerate启动进行多机多卡训练。

首先运行以下命令，根据需求回答一系列问题后生成配置文件：

.. code-block:: bash

    accelerate config

下面提供一个示例配置文件：

.. code-block:: yaml

    # accelerate_multiNode_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: true
    distributed_type: MULTI_GPU
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    gpu_ids: all
    machine_rank: 0
    main_process_ip: '192.168.0.1'
    main_process_port: 29500
    main_training_function: main
    mixed_precision: fp16
    num_machines: 2
    num_processes: 16
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false


您可以通过运行以下指令开始训练:

.. code-block:: bash

    accelerate launch \
    --config_file accelerate_multiNode_config.yaml \
    train.py llm_config.yaml

.. _fsdp:

fsdp
~~~~~~~~~~~~~~~~~~~~~~~~~


.. _fsdp单机多卡:

.. _fsdp多机多卡:


PyTorch的全切片数据并行技术（Fully Sharded Data Parallel）能让我们处理更多更大的模型。LLaMA-Factory支持使用fsdp引擎进行分布式训练。


llamafactory-cli
+++++++++++++++++++++++++

您只需根据需要修改 ``examples/accelerate/fsdp_config.yaml`` 以及 ``examples/extras/fsdp_qlora/llama3_lora_sft.yaml`` ，文件然后运行以下命令即可启动fsdp+QLoRA微调：

.. code-block:: bash

    bash examples/extras/fsdp qlora/train.sh



accelerate
++++++++++++++++++++



此外，您也可以使用accelerate启动fsdp引擎， **节点数与GPU数可以通过 num_machines 和  num_processes 指定**。对此，Huggingface提供了便捷的配置功能。
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


随后，您可以使用以下命令启动训练：

.. code-block:: bash

    accelerate launch \
    --config_file fsdp_config.yaml \
    train.py llm_config.yaml

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

llamafactory-cli
*********************

您可以使用llamafactory-cli启动DeepSpeed引擎进行单机多卡训练。

.. code-block:: bash

    llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml


deepspeed
**************************

您也可以使用deepspeed指令启动DeepSpeed引擎进行单机多卡训练。

.. code-block:: bash

    deepspeed --include localhost:1 your_program.py <normal cl args> --deepspeed ds_config.json


.. note:: 

    使用deepspeed指令启动DeepSpeed引擎时您无法使用 ``CUDA_VISIBLE_DEVICES`` 指定GPU。而需要：

    .. code-block:: bash

        deepspeed --include localhost:1 your_program.py <normal cl args> --deepspeed ds_config.json
    
    ``--include localhost:1`` 表示只是用本节点的gpu1。

.. _deepspeed多机多卡:

多机多卡
+++++++++++++++++++++


LLaMA-Factory支持使用deepspeed的多机多卡训练，您可以通过以下命令启动：

.. code-block:: bash

    FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml
    FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml


deepspeed
******************************

您也可以使用 ``deepspeed`` 命令来启动多机多卡训练。

.. code-block:: bash

    deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
    your_program.py <normal cl args> --deepspeed ds_config.json



.. note::

    * 关于hostfile:
        hostfile的每一行指定一个节点，每行的格式为 ``<hostname> slots=<num_slots>`` ，
        其中 ``<hostname>`` 是节点的主机名， ``<num_slots>`` 是该节点上的GPU数量。下面是一个例子：
        .. code-block:: 

            worker-1 slots=4
            worker-2 slots=4

        请在 `https://www.deepspeed.ai/getting-started/ <https://www.deepspeed.ai/getting-started/>`_ 了解更多。
    
    * 如果没有指定 ``hostfile`` 变量,DeepSpeed会搜索 ``/job/hostfile`` 文件。如果仍未找到，那么DeepSpeed会使用本机上所有可用的GPU。

accelerate
*******************
您还可以使用accelerate启动deepspeed引擎。
首先通过以下命令生成deepspeed配置文件：
.. code-block:: bash

    accelerate config

下面提供一个配置文件示例：

.. code-block:: yaml

    # deepspeed_config.yaml
    compute_environment: LOCAL_MACHINE
    debug: false
    deepspeed_config:
        deepspeed_multinode_launcher: standard
        gradient_accumulation_steps: 8
        offload_optimizer_device: none
        offload_param_device: none
        zero3_init_flag: false
        zero_stage: 3
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    machine_rank: 0
    main_process_ip: '192.168.0.1'
    main_process_port: 29500
    main_training_function: main
    mixed_precision: fp16
    num_machines: 2
    num_processes: 16
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false

随后，您可以使用以下命令启动训练：

.. code-block:: bash

    accelerate launch \
    --config_file deepspeed_config.yaml \
    train.py llm_config.yaml



deepspeed配置文件
++++++++++++++++++++++

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

