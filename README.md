# Custom Input Data

对TPA-LSTM代码工程化封装使用的尝试。

# TPA-LSTM

Forked from [this repo](https://github.com/gantheory/TPA-LSTM): Original Implementation of [''Temporal Pattern Attention for Multivariate Time Series Forecasting''](https://arxiv.org/abs/1809.04206).

## Dependencies

参照[原repo](https://github.com/gantheory/TPA-LSTM#dependencies)，如果不使用music相关的数据集（e.g. 原论文的`lpd5`或`muse`）则无需`pypianoroll`，使用conda即可导入requirements环境。

```
conda install --file requirements.txt
```


## Usage

参照[demo_test.py](demo_test.py),在atom编辑器使用[hydrogen](https://atom.io/packages/hydrogen)可逐步分条执行指令或导出Jupyter Notebook

## Todo List

- [x] 自定义数据文件作为模型的输入
  - [x] 增加参数`custom`标记用户自定义数据，`dataset_address`为`.parquet`格式的MTS数据在本地的地址，其中`'date'`字段为时间，`split_date`为划分`['train','test','valid']`依据的时间点
  - [x] 实现`customDataGenerator`类用于自定义数据导入
  - [x] 模型成功运行并log输出误差

- [ ] 修正多次在本地导出数据以及其导致的重复log问题
- [x] 恢复出预测结果的原始数据并存储
  - [x] 预测结果（包含时间戳）保存为`<output_dir>/<data_set>_predict_output.parquet`文件
  - [ ] 简化本地目录读写，在序列化过程中关联时间戳
- [ ] 重新拆分建模逻辑，减少耦合
