评估
=========================

通用能力评估
-----------------------

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
    task: mmlu_test # mmlu_test, ceval_validation, cmmlu_test
    template: fewshot
    lang: en
    n_shot: 5

    ### output
    save_dir: saves/llama3-8b/lora/eval

    ### eval
    batch_size: 4


NLG 评估
--------------------------

此外，您还可以通过 ``llamafactory-cli train examples/extras/nlg_eval/llama3_lora_predict.yaml`` 来获得模型的 BLEU 和 ROUGE 分数以评价模型生成质量。

配置示例文件 ``examples/extras/nlg_eval/llama3_lora_predict.yaml`` 具体如下：

.. code-block:: yaml

      ### examples/extras/nlg_eval/llama3_lora_predict.yaml
      ### model
      model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
      adapter_name_or_path: saves/llama3-8b/lora/sft

      ### method
      stage: sft
      do_predict: true
      finetuning_type: lora

      ### dataset
      eval_dataset: identity,alpaca_en_demo
      template: llama3
      cutoff_len: 2048
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

同样，您也通过在指令 ``python scripts/vllm_infer.py --model_name_or_path path_to_merged_model --dataset alpaca_en_demo`` 中指定模型、数据集以使用 vllm 推理框架以取得更快的推理速度。



评估相关参数
-------------------------

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