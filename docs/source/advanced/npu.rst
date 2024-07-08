华为NPU适配
================

目前LLaMA-Factory 通过torch-npu库完成了对华为昇腾910b系列芯片的支持, 包含32GB和64GB两个版本。跟其他使用相比，会需要额外3个前置条件

1. 加速卡本身的驱动正常安装
#. 昇腾npu计算相关 Toolkit 和 Kernels库正常安装
#. torch-npu库正常安装

另外python版本建议使用3.10， 目前该版本对于npu的使用情况会相对稳定，其他版本可能会遇到一些不确定的情况

依赖1
----------
依赖1一般在机器交付的时候，工程师会一起完成安装，使用``npu-smi info``验证如下
.. image:: ../assets/advanced/npu-smi.png

依赖2
----------
依赖2的安装方式有两种
1. 在裸金属环境下自主安装相关开发环境

.. list-table:: 相关包建议版本
   :widths: 30 10 60
   :header-rows: 1

   * - Requirement
     - Minimum
     - Recommend
   * - CANN
     - 8.0.RC1
     - 8.0.RC1
   * - torch
     - 2.1.0
     - 2.1.0
   * - torch-npu
     - 2.1.0
     - 2.1.0.post3
   * - deepspeed
     - 0.13.2
     - 0.13.2

可以按照`官方指导 <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/softwareinstall/instg/atlasdeploy_03_0031.html>`_ 或者使用以下命令完成安装
.. code-block:: bash
    # replace the url according to your CANN version and devices
    # install CANN Toolkit
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC701/Ascend-cann-toolkit_8.0.RC1.alpha001_linux-"$(uname -i)".run
    bash Ascend-cann-toolkit_8.0.RC1.alpha001_linux-"$(uname -i)".run --install

    # install CANN Kernels
    wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC701/Ascend-cann-kernels-910b_8.0.RC1.alpha001_linux.run
    bash Ascend-cann-kernels-910b_8.0.RC1.alpha001_linux.run --install

    # set env variables
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

#. 使用华为昇腾团队一起提供的docker镜像
镜像使用方法和下载方式请参考 `32GB <http://mirrors.cn-central-221.ovaijisuan.com/detail/130.html>`_ 和 `64GB <http://mirrors.cn-central-221.ovaijisuan.com/detail/131.html>`_


依赖3
----------
依赖3建议在安装llama-factory的时候一起选配安装， 把 ``torch-npu``一起加入安装目标，命令如下
.. code-block:: bash
    pip install -e ".[torch-npu,metrics]"

依赖校验
----------
3个依赖都安装后，可以通过如下的python脚本对``torch_npu``的可用情况做一下校验
.. code-block:: python
    import torch
    import torch_npu
    print(torch.npu.is_available())

预期结果是打印true
.. image:: ../assets/advanced/npu-torch.png

开始使用npu
----------
前面依赖安装完毕和完成校验后，即可像文档的其他部分一样正常使用 llamafactory-cli 的相关功能，主要的区别是需要修改一下 可用设备的变量使用
将原来的Nvidia卡的变量``CUDA_VISIBLE_DEVICES`` 替换为 ``ASCEND_RT_VISIBLE_DEVICES``， 类似如下命令
.. code-block:: bash
    ASCEND_RT_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

.. note::
    如果在推理过程中遇到了长时间卡顿或者其他错误，请尝试在``llamafactory-cli`` 的配置参数中指定 ``do_sample`` 为 ``false`` 即可，比如在yaml中修改

    .. code-block:: yaml
        do_sample: false