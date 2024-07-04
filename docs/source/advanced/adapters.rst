调优算法
=============

LLaMA-Factory支持多种调优算法，包括： :ref:`Full Parameter Fine-tuning <full>` 、 :ref:`Freeze <Freeze>` 、 :ref:`LoRA <LoRA>` 、 :ref:`Galore <Galore>` 、 :ref:`BAdam <BAdam>` 。

.. .....


.. list-table::
    :widths: 30 20 10
    :header-rows: 1

    * - 微调方法
      - 所需显存
      - 收敛速度
    * - Full Parameter Fine-tuning
      - ..
      - ..
    * - Freeze
      - ..
      - ..
    * - LoRA
      - ..
      - ..
    * - Galore
      - ..
      - ..
    * - BAdam
      - ..
      - ..


.. _full:

Full Parameter Fine-tuning
--------------------
全参微调指的是在训练过程中对于预训练模型的所有权重都进行更新，但其对显存的要求是巨大的。

.. 以 <TODO> 为例子

如果您需要进行全参微调，请将 ``finetuning_type`` 设置为 ``full`` 。
下面是一个例子：

.. code-block:: yaml

    ### examples/train_full/llama3_full_sft_ds3..yaml
    # ...
    finetuning_type: full
    # ...
    # 如果需要使用deepspeed:
    deepspeed: examples/deepspeed/ds_z3_config.json

.. _freeze:

Freeze
--------------------------

Freeze(冻结微调)指的是在训练过程中只对模型的小部分权重进行更新，这样可以降低对显存的要求。

.. <以..>

如果您需要进行冻结微调，请将 ``finetuning_type`` 设置为 ``freeze`` 并且设置相关参数,
例如冻结的层数 ``freeze_trainable_layers`` 、可训练的模块名称 ``freeze_trainable_modules`` 等。



以下是一个例子：

.. code-block:: yaml

    ### model
    model_name_or_path: models/llama3-8b-instruct-pro

    ### method
    stage: sft
    do_train: true
    finetuning_type: freeze
    freeze_trainable_layers: 8
    freeze_trainable_modules: all
    use_llama_pro: true

    ### dataset
    dataset: identity,alpaca_en_demo
    template: llama3
    cutoff_len: 1024
    max_samples: 1000
    overwrite_cache: true
    preprocessing_num_workers: 16

    ### output
    output_dir: saves/llama3-8b-instruct-pro/freeze/sft
    logging_steps: 10
    save_steps: 500
    plot_loss: true
    overwrite_output_dir: true

    ### train
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 1.0e-4
    num_train_epochs: 3.0
    lr_scheduler_type: cosine
    warmup_ratio: 0.1
    bf16: true
    ddp_timeout: 180000000

    ### eval
    val_size: 0.1
    per_device_eval_batch_size: 1
    eval_strategy: steps
    eval_steps: 500


.. list-table:: FreezeArguments
   :widths: 30 10 50
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
   * - freeze_trainable_layers
     - int
     - 可训练层的数量。正数表示最后 n 层被设置为可训练的，负数表示前 n 层被设置为可训练的。默认值为 ``2``
   * - freeze_trainable_modules
     - str
     - 可训练层的名称。使用 ``all`` 来指定所有模块。默认值为 ``all``
   * - freeze_extra_modules[非必须]
     - str
     - 除了隐藏层外可以被训练的模块名称，被指定的模块将会被设置为可训练的。使用逗号分隔多个模块。默认值为 ``None``


.. _LoRA:

LoRA
--------------------------
如果您需要进行LoRA微调，请将 ``finetuning_type`` 设置为 ``lora`` 并且设置相关参数。
下面是一个例子：

.. code-block:: yaml

    ...
    ### method
    stage: sft
    do_train: true
    finetuning_type: lora
    lora_target: all
    lora_rank: 8
    lora_alpha: 16
    lora_dropout: 0.1
    ...


.. list-table:: LoraArguments
   :widths: 30 10 50
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
   * - additional_target[非必须]
     - [str,]
     - 除LoRA层之外设置为可训练并保存在最终检查点中的模块名称。使用逗号分隔多个模块。默认值为 ``None``
   * - lora_alpha[非必须]
     - int
     - LoRA 缩放系数。一般情况下为 lora_rank * 2, 默认值为 ``None``
   * - lora_dropout
     - float
     - LoRA微调中的dropout率。默认值为 ``0``
   * - lora_rank
     - int
     - LoRA微调的本征维数 ``r``， ``r`` 越大可训练的参数越多。默认值为 ``8``
   * - lora_target
     - str
     - 应用LoRA方法的模块名称。使用逗号分隔多个模块，使用 ``all`` 指定所有模块。默认值为 ``all``
   * - loraplus_lr_ratio[非必须]
     - float
     - LoRA+学习率比例(``r = ηA/ηB``)。 ``ηA, ηB`` 分别是 adapter matrices A 与 B 的学习率。实验表明，将这个值设置为 ``16`` 会取得较好的初始结果。当任务较为复杂时需要将这个值设置得大一些。默认值为 ``None``
   * - loraplus_lr_embedding[非必须]
     - float
     - LoRA+嵌入层的学习率, 默认值为 ``1e-6``
   * - use_rslora
     - bool
     - 是否使用秩稳定LoRA(Rank-Stabilized LoRA)，默认值为 ``False``
   * - use_dora
     - bool
     - 是否使用权重分解LoRA（Weight-Decomposed LoRA），默认值为 ``False``
   * - pissa_init
     - bool
     - 是否初始化PiSSA适配器，默认值为 ``False``
   * - pissa_iter
     - int
     - PiSSA中FSVD执行的迭代步数。使用 ``-1`` 将其禁用，默认值为 ``16``
   * - pissa_convert
     - bool
     - 是否将PiSSA适配器转换为正常的LoRA适配器，默认值为 ``False``
   * - create_new_adapter
     - bool
     - 是否创建一个具有随机初始化权重的新适配器，默认值为 ``False``





.. _Galore:

Galore
------------------------

当您需要在训练中使用 GaLore（Gradient Low-Rank Projection）算法时，可以通过设置 ``GaloreArguments`` 中的参数进行配置。

.. warning:: 

  不要将LoRA和GaLore/BAdam 一起使用。

下面是一个例子：

.. code-block:: yaml

    ...
    ### method
    stage: sft
    do_train: true
    finetuning_type: full
    use_galore: true
    galore_layerwise: true
    galore_target: mlp,self_attn
    galore_rank: 128
    galore_scale: 2.0
    ...
    



.. list-table:: GaLoreArguments
   :widths: 30 10 60
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
   * - use_galore
     - bool
     - 是否使用 GaLore 算法，默认值为 ``False``。
   * - galore_target
     - str
     - 应用 GaLore 的模块名称。使用逗号分隔多个模块，使用 ``all`` 指定所有线性模块。默认值为 ``all``。
   * - galore_rank
     - int
     - GaLore 梯度的秩，默认值为 ``16``。
   * - galore_update_interval
     - int
     - 更新 GaLore 投影的步数间隔，默认值为 ``200``。
   * - galore_scale
     - float
     - GaLore 的缩放系数，默认值为 ``0.25``。
   * - galore_proj_type
     - Literal
     - GaLore 投影的类型，可选值有： ``std`` , ``reverse_std``, ``right``, ``left``, ``full``。默认值为 ``std``。
   * - galore_layerwise
     - bool
     - 是否启用逐层更新以进一步节省内存，默认值为 ``False``。



.. _BAdam:

BAdam
-------------------------

.. warning:: 

  不要将LoRA和GaLore/BAdam 一起使用。

BAdam是一种内存高效的全参优化方法，您通过配置 ``BAdamArgument`` 中的参数可以对其进行详细设置。
下面是一个例子：

.. code-block:: yaml

    ### model
    ...
    ### method
    stage: sft
    do_train: true
    finetuning_type: full
    use_badam: true
    badam_mode: layer
    badam_switch_mode: ascending
    badam_switch_interval: 50
    badam_verbose: 2
    pure_bf16: true
    ...

.. warning:: 

  使用BAdam时请设置 ``finetuning_type`` 为 ``full`` 且 ``pure_bf16`` 为 ``True`` 。

.. list-table:: BAdamArgument
   :widths: 30 10 60
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
   * - use_badam
     - bool
     - 是否使用 BAdam 优化器，默认值为 ``False``。
   * - badam_mode
     - Literal
     - BAdam 的使用模式，可选值为 ``layer`` 或 ``ratio``，默认值为 ``layer``。
   * - badam_start_block
     - Optional[int]
     - layer-wise BAdam 的起始块索引，默认值为 ``None``。
   * - badam_switch_mode
     - Optional[Literal]
     - layer-wise BAdam 中块更新策略，可选值有： ``ascending``, ``descending``, ``random``, ``fixed``。默认值为 ``ascending``。
   * - badam_switch_interval
     - Optional[int]
     - layer-wise BAdam 中块更新步数间隔。使用 ``-1`` 禁用块更新，默认值为 ``50``。
   * - badam_update_ratio
     - float
     - ratio-wise BAdam 中的更新比例，默认值为 ``0.05``。
   * - badam_mask_mode
     - Literal
     - BAdam 优化器的掩码模式，可选值为 ``adjacent`` 或 ``scatter``，默认值为 ``adjacent``。
   * - badam_verbose
     - int
     - BAdam 优化器的详细输出级别，0 表示无输出，1 表示输出块前缀，2 表示输出可训练参数。默认值为 ``0``。


