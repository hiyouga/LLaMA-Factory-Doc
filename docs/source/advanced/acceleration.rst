加速
=====================

LLaMA-Factory 支持多种加速技术，包括：:ref:`FlashAttention <flashattn>` 、 :ref:`Unsloth <sloth>`  。




.. _flashattn:


FlashAttention
----------------------------

`FlashAttention  <https://github.com/Dao-AILab/flash-attention/>`_ 能够加快注意力机制的运算速度，同时减少对内存的使用。

如果您想使用 FlashAttention,请在启动训练时在训练配置文件中添加以下参数：

.. code-block:: yaml 

    flash_attn: fa2



.. _sloth:

Unsloth
---------------------------

`unsloth <https://github.com/unslothai/unsloth/>`_ 框架支持 Llama, Mistral, Phi-3, Gemma, Yi, DeepSeek, Qwen等大语言模型并且支持4bit和16bit的QLoRA/LoRA微调，该框架在提高运算速度的同时还减少了显存占用。

如果您想使用 Unsloth,请在启动训练时在训练配置文件中添加以下参数：

.. code-block:: yaml

    use_unsloth: True 


