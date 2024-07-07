评估
=========================

在完成模型训练后，您可以通过 ``llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml`` 来评估模型效果。

配置示例如下：

.. code-block:: yaml

    ### examples/train_lora/llama3_lora_eval.yaml
    ### model
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    adapter_name_or_path: saves/llama3-8b/lora/sft

    ### method
    finetuning_type: lora

    ### dataset
    task: mmlu
    split: test
    template: fewshot
    lang: en
    n_shot: 5

    ### output
    save_dir: saves/llama3-8b/lora/eval

    ### eval
    batch_size: 4



在 :ref:`批量推理` 的过程中，模型的 BLEU 和 ROUGE 分数会被自动计算并保存，您也可以通过此方法评估模型。
