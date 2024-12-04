评估
=========================

在完成模型训练后，您可以通过 ``llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml`` 来评估模型效果。

配置示例文件 ``examples/train_lora/llama3_lora_eval.yaml`` 具体如下：

.. code-block:: yaml

    ### examples/train_lora/llama3_lora_eval.yaml
    ### model
    model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
    adapter_name_or_path: saves/llama3-8b/lora/sft # 可选项

    ### method
    finetuning_type: lora

    ### dataset
    task: mmlu_test
    template: fewshot
    lang: en
    n_shot: 5

    ### output
    save_dir: saves/llama3-8b/lora/eval

    ### eval
    batch_size: 4



在 :ref:`批量推理` 的过程中，模型的 BLEU 和 ROUGE 分数会被自动计算并保存，您也可以通过此方法评估模型。


下面是评估相关参数的介绍:

.. list-table:: EvalArguments
   :widths: 10 10 40
   :header-rows: 1

   * - 参数名称
     - 类型
     - 介绍
   * - task
     - str
     - 评估任务的名称，可选项有 mmlu_test, ceval_validation, cmmlu_test
   * - task_dir
     - str
     - 包含评估数据集的文件夹路径，默认值为 ``evaluation``。
   * - batch_size
     - int
     - 每个GPU使用的批量大小，默认值为 ``4``。
   * - seed
     - int
     - 用于数据加载器的随机种子，默认值为 ``42``。
   * - lang
     - str
     - 评估使用的语言，可选值为 ``en``、 ``zh``。默认值为 ``en``。
   * - n_shot
     - int
     - few-shot 的示例数量，默认值为 ``5``。
   * - save_dir
     - str
     - 保存评估结果的路径，默认值为 ``None``。 如果该路径已经存在则会抛出错误。
   * - download_mode
     - str
     - 评估数据集的下载模式，默认值为 ``DownloadMode.REUSE_DATASET_IF_EXISTS``。如果数据集已经存在则重复使用，否则则下载。