数据处理
============================

`dataset_info.json <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json/>`_ 包含了所有可用的数据集。如果您希望使用自定义数据集，请 **务必** 在 ``dataset_info.json`` 文件中添加数据集描述，并通过修改 ``dataset: 数据集名称`` 配置来使用数据集。

目前我们支持 :ref:`alpaca<alpaca>` 格式和  :ref:`sharegpt<Sharegpt>` 格式的数据集



.. _alpaca: 

Alpaca
------------------

针对不同任务，数据集格式要求如下：

* :ref:`指令监督微调 <指令监督微调数据集>`
* :ref:`预训练 <预训练数据集>`
* :ref:`偏好训练 <偏好数据集-1>`
* :ref:`KTO <KTO数据集>`
* :ref:`多模态 <多模态数据集>`

.. _指令监督微调数据集:

指令监督微调数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**样例数据集**： `指令监督微调样例数据集 <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_zh_demo.json/>`_

在进行指令监督微调时， ``instruction`` 列对应的内容会与 ``input`` 列对应的内容拼接后作为人类指令，即人类指令为 ``instruction\ninput``。而 ``output`` 列对应的内容为模型回答。

如果指定， ``system`` 列对应的内容将被作为系统提示词。

``history`` 列是由多个字符串二元组构成的列表，分别代表历史消息中每轮对话的指令和回答。注意在指令监督微调时，历史消息中的回答内容也会被用于模型学习。


.. code-block:: json

  [
    {
      "instruction": "人类指令（必填）",
      "input": "人类输入（选填）",
      "output": "模型回答（必填）",
      "system": "系统提示词（选填）",
      "history": [
        ["第一轮指令（选填）", "第一轮回答（选填）"],
        ["第二轮指令（选填）", "第二轮回答（选填）"]
      ]
    }
  ]
  [
    {
      "instruction": "人类指令（必填）",
      "input": "人类输入（选填）",
      "output": "模型回答（必填）",
      "system": "系统提示词（选填）",
      "history": [
        ["第一轮指令（选填）", "第一轮回答（选填）"],
        ["第二轮指令（选填）", "第二轮回答（选填）"]
      ]
    }
  ]


对于上述格式的数据， ``dataset_info.json`` 中的 **数据集描述** 应为：

.. code-block:: json

  "数据集名称": {
    "file_name": "data.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system",
      "history": "history"
    }
  }

.. _预训练数据集:

预训练数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**样例数据集**： `预训练样例数据集 <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/c4_demo.json/>`_

在预训练时，只有 ``text`` 列中的内容会用于模型学习。

.. code-block:: json

  [
    {"text": "document"},
    {"text": "document"}
  ]

对于上述格式的数据， ``dataset_info.json`` 中的 **数据集描述** 应为：

.. code-block:: json

  "数据集名称": {
    "file_name": "data.json",
    "columns": {
      "prompt": "text"
    }
  }


.. _偏好数据集-1:
偏好数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


**样例数据集**： `偏好样例数据集 <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dpo_zh_demo.json/>`_

偏好数据集用于奖励模型训练、DPO 训练和 ORPO 训练。

它需要在 ``chosen`` 列中提供更优的回答，并在 ``rejected`` 列中提供更差的回答。

.. code-block:: json

  [
    {
      "instruction": "人类指令（必填）",
      "input": "人类输入（选填）",
      "chosen": "优质回答（必填）",
      "rejected": "劣质回答（必填）"
    }
  ]

对于上述格式的数据，``dataset_info.json`` 中的 **数据集描述** 应为：

.. code-block:: json

  "数据集名称": {
    "file_name": "data.json",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }

.. _KTO数据集:
KTO 数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**样例数据集**： `KTO样例数据集 <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/kto_en_demo.json/>`_

KTO 数据集需要额外添加一个 ``kto_tag`` 列，包含 ``bool`` 类型的人类反馈。

.. code-block:: json

  [
    {
      "instruction": "人类指令（必填）",
      "input": "人类输入（选填）",
      "output": "模型回答（必填）",
      "kto_tag": "人类反馈 [true/false]（必填）"
    }
  ]



对于上述格式的数据， ``dataset_info.json`` 中的 **数据集描述** 应为：

.. code-block:: json

  "数据集名称": {
    "file_name": "data.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "kto_tag": "kto_tag"
    }
  }


.. _多模态数据集:

多模态数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**样例数据集**： `多模态样例数据集 <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/mllm_demo.json/>`_

多模态数据集需要额外添加一个 ``images`` 列，包含输入图像的路径。目前我们仅支持单张图像输入。

.. code-block:: json

  [
    {
      "instruction": "人类指令（必填）",
      "input": "人类输入（选填）",
      "output": "模型回答（必填）",
      "images": [
        "图像路径（必填）"
      ]
    }
  ]

对于上述格式的数据， ``dataset_info.json`` 中的 **数据集描述** 应为：

.. code-block:: json

  "数据集名称": {
    "file_name": "data.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "images": "images"
    }
  }

.. _Sharegpt:

Sharegpt
------------------------------------------

针对不同任务，数据集格式要求如下：

* :ref:`指令监督微调 <指令监督微调数据集-2>`
* :ref:`偏好训练 <偏好数据集-2>`
* :ref:`OpenAI格式 <OpenAI格式>`

.. note::
  * sharegpt 格式中的 KTO 数据集和多模态数据集与 alpaca 格式的类似。
  * 预训练数据集不支持 sharegpt 格式。



.. _指令监督微调数据集-2:
指令监督微调数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


**样例数据集**： `指令监督微调样例数据集 <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/glaive_toolcall_zh_demo.json/>`_

相比 ``alpaca`` 格式的数据集， ``sharegpt`` 格式支持 **更多的角色种类**，例如 human、gpt、observation、function 等等。它们构成一个对象列表呈现在 ``conversations`` 列中。

注意其中 human 和 observation 必须出现在奇数位置，gpt 和 function 必须出现在偶数位置。


.. code-block:: json

  [
    {
      "conversations": [
        {
          "from": "human",
          "value": "人类指令"
        },
        {
          "from": "function_call",
          "value": "工具参数"
        },
        {
          "from": "observation",
          "value": "工具结果"
        },
        {
          "from": "gpt",
          "value": "模型回答"
        }
      ],
      "system": "系统提示词（选填）",
      "tools": "工具描述（选填）"
    }
  ]



对于上述格式的数据， ``dataset_info.json`` 中的 **数据集描述** 应为：


.. code-block:: json

  "数据集名称": {
    "file_name": "data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system",
      "tools": "tools"
    }
  }


.. _偏好数据集-2:
偏好数据集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


**样例数据集**： `偏好数据样例数据集 <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dpo_zh_demo.json/>`_


Sharegpt 格式的偏好数据集同样需要在 ``chosen`` 列中提供更优的消息，并在 ``rejected`` 列中提供更差的消息。

.. code-block:: json

  [
    {
      "conversations": [
        {
          "from": "human",
          "value": "人类指令"
        },
        {
          "from": "gpt",
          "value": "模型回答"
        },
        {
          "from": "human",
          "value": "人类指令"
        }
      ],
      "chosen": {
        "from": "gpt",
        "value": "优质回答"
      },
      "rejected": {
        "from": "gpt",
        "value": "劣质回答"
      }
    }
  ]

对于上述格式的数据，``dataset_info.json`` 中的 **数据集描述** 应为：

.. code-block:: json

  "数据集名称": {
    "file_name": "data.json",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }

.. _OpenAI格式:
OpenAI格式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenAI 格式仅仅是 ``sharegpt`` 格式的一种特殊情况，其中第一条消息可能是系统提示词。

.. code-block:: json

  [
    {
      "messages": [
        {
          "role": "system",
          "content": "系统提示词（选填）"
        },
        {
          "role": "user",
          "content": "人类指令"
        },
        {
          "role": "assistant",
          "content": "模型回答"
        }
      ]
    }
  ]



对于上述格式的数据， ``dataset_info.json`` 中的 **数据集描述** 应为：

.. code-block:: json

  "数据集名称": {
    "file_name": "data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
