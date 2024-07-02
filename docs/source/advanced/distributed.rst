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


