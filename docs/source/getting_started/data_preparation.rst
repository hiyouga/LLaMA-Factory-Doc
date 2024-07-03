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

指令监督微调(Instruct Tuning)通过让模型学习详细的指令以及对应的回答来优化模型在特定指令下的表现。


``instruction`` 列对应的内容为人类指令， ``input`` 列对应的内容为人类输入， ``output`` 列对应的内容为模型回答。下面是一个例子

.. code-block:: json

  "alpaca_zh_demo.json"
  {
    "instruction": "计算这些物品的总费用。 ",
    "input": "输入：汽车 - $3000，衣服 - $100，书 - $20。",
    "output": "汽车、衣服和书的总费用为 $3000 + $100 + $20 = $3120。"
  },


在进行指令监督微调时， ``instruction`` 列对应的内容会与 ``input`` 列对应的内容拼接后作为最终的人类输入，即人类输入为 ``instruction\ninput``。而 ``output`` 列对应的内容为模型回答。
在上面的例子中，人类的最终输入是：

.. code-block:: text

  计算这些物品的总费用。
  输入：汽车 - $3000，衣服 - $100，书 - $20。

模型的回答是：

.. code-block:: text

  汽车、衣服和书的总费用为 $3000 + $100 + $20 = $3120。



如果指定， ``system`` 列对应的内容将被作为系统提示词。

``history`` 列是由多个字符串二元组构成的列表，分别代表历史消息中每轮对话的指令和回答。注意在指令监督微调时，历史消息中的回答内容也会被用于模型学习。

指令监督微调数据集 **格式要求** 如下：

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


大语言模型通过学习未被标记的文本进行预训练，从而学习语言的表征。
预训练数据集文本描述格式如下：

.. code-block:: json

  [
    {"text": "document"},
    {"text": "document"}
  ]

在预训练时，只有 ``text`` 列中的 **内容** （即document）会用于模型学习。

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


偏好数据集用于奖励模型训练、DPO 训练和 ORPO 训练。对于系统指令和人类输入，偏好数据集给出了一个更优的回答和一个更差的回答。
以下是一个例子：

.. code-block:: json

  {
    "conversations": [
      {
        "from": "system",
        "value": "您是一名AI助手。用户会给您一个任务。您的目标是尽量忠实地完成任务。在执行任务时，要逐步思考并合理地解释您的步骤。"
      },
      {
        "from": "human",
        "value": "体重增加是一个复杂的问题。但毫无疑问，西方国家当前的食物环境——大量含有高度加工的咸、甜和油脂食品——起着重要作用。我们饮食中摄入的大部分盐都是在制造过程中添加的。这与大多数食物中天然存在的少量盐或在餐桌上或在家里烹饪时添加的盐形成对比。盐可以增加食物的可口性，帮助保鲜；这就是为什么它通常被添加到加工食品中。盐和脂肪的结合被认为是促进食物被 passively 过量消费的强大组合。想想看，咸薯片有多令人回味。对咸和油脂食物更喜欢与整体摄入更多千焦耳、不受控制的饮食和儿童超重有关。食物过量摄入与其盐度程度之间的联系被认为比对甜和油脂食物的喜好更强。从进一步揭示盐可能如何影响对油脂食物的过量摄入，Deakin 大学的感官研究人员招募了 48 名健康成年人参加品尝小组。在四个午餐时间段（在那天早上进行了标准早餐后），每个人都吃了一顿经过脂肪和盐含量调整的通心粉奶酪餐。四种餐食组合分别是低脂/低盐、低脂/高盐、高脂/低盐和高脂/高盐。鼓励参与者吃到饱为止。还记录了进食速度、餐食的可口性以及主观评价的饥饿感和饱腹感。盐促进过量进食当食用含盐量高的餐食时，摄入的千焦耳增加了 11%。而这与脂肪含量高低无关。餐食的脂肪含量并没有导致人们按重量摄入更多食物，但由于其更高的能量密度，这意味着摄入了更多千焦耳。问题：那些习惯于喜欢高热量密度饮食的人可能经常食用高盐食品吗？=== 上述问题的答案是\n可选项：\n(1). 一周以上。\n(2). 大多数时候通心粉奶酪被提供时。\n(3). 没有足够的信息\n(4). 通常在午餐时间。\n答案："
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "(3). 不够信息"
    },
    "rejected": {
      "from": "gpt",
      "value": "根据文本提供的信息，对于问题“那些习惯于偏好高热量饮食的人通常多久吃一次高盐食品？”的答案是：\n\n（1）一周以上。\n\n文本指出，高盐餐会导致千焦耳的过度摄入增加，无论餐食的脂肪含量如何。这表明那些习惯于偏好高热量饮食的人可能更有可能频繁地消费高盐食品，因为他们在有机会时更有可能过度摄入千焦耳。因此，选项（1）是最佳答案。"
    }
  }


`一些研究 <https://openai.com/index/instruction-following/>`_ 表明通过让模型学习“什么更好”可以使得模型更加迎合人类的需求。
甚至可以使得参数相对较少的模型的表现优于参数更多的模型。


偏好数据集需要在 ``chosen`` 列中提供更优的回答，并在 ``rejected`` 列中提供更差的回答，在一轮问答中其格式如下：

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

KTO数据集与偏好数据集类似，但不同于给出一个更优的回答和一个更差的回答，KTO数据集对每一轮问答只给出一个 true/false 的 ``label``。

除了 ``instruction`` 以及 ``input`` 组成的人类最终输入和模型回答 ``output`` ，KTO 数据集还需要额外添加一个 ``kto_tag`` 列（true/false）来表示人类的反馈。

在一轮问答中其格式如下：
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
