# t2t针对自定义平行语料的配置
## 准备
- 文件组织格式如下所示：
![]
(https://github.com/ruxintan/t2t/blob/master/1530325487479.jpg)
- rawdata:存放训练集和验证集；训练集命名为：xx.txt;验证集命名为：dev.xx.xx代表语言缩写。
- decoder：存放最后的测试结果（初始化为空）
- self_data:存放t2t编译之后的二进制数据
- self_script：存放编写的problems文件
- train：训练相关数据

## 开始
1. 将语料库按照训练集、验证集、测试集命名规则命名（参考准备中的rawdata）
2. 训练集打包xx-xx-train.zip，验证集打包xx-xx-dev.zip，同时放进rawdata中；测试集放入decoder中(命名为test.xx)
3. 配置probelms文件：


```
_NC_TRAIN_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/vi-zh-train.zip", [    #训练集压缩包
        "vi.txt",  # 训练集              
        "zh.txt"   # 训练集
    ]
]]
# Test set from News Commentary. 2000 lines
_NC_TEST_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/vi-zh-dev.zip",		#验证集压缩包
    ("dev.vi", "dev.zh")。#验证集
]]
```
## t2t训练
### 预处理
- 
```
CUDA_VISIBLE_DEVICES=0 t2t-datagen  --t2t_usr_dir=self_script --data_dir=self_data --tmp_dir=rawdata --problem=my_problem
```

### 训练
- 
```
CUDA_VISIBLE_DEVICES=0 t2t-trainer --t2t_usr_dir=self_script --problem=my_problem --data_dir=self_data --model=transformer --hparams_set=transformer_base_single_gpu --output_dir=train
```

### 解码
-
```
CUDA_VISIBLE_DEVICES=0 t2t-decoder --t2t_usr_dir=self_script --problem=my_problem --data_dir=self_data --model=transformer --hparams_set=transformer_base_single_gpu --output_dir=train --decode_hparams="beam_size=4,alpha=0.6" --decode_from_file=decoder/test.uy --decode_to_file=decoder/result.ch
```
