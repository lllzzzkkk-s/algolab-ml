<<<<<<< HEAD
# algolab-ml
for Algorithm
=======
# Algolab ML 模块化骨架

依据你上传的笔记自动生成，覆盖：赛题分析→数据清洗→特征工程→模型选择→模型融合→时间序列→Word2Vec。

## 快速开始
```bash
pip install -e .
python cli/mlrun.py --csv path/to/train.csv --target label --model xgb
```

## 目录说明
- ml/data: 读写、清洗、基础拆分与问题类型判断
- ml/features: 表格预处理、时间序列构造、Word2Vec
- ml/models: 模型仓库、交叉验证、训练、融合
- ml/eval: 指标集合
- ml/pipelines: 端到端表格数据管线
- cli/mlrun.py: 命令行入口

> 生成时间：2025-08-26T07:02:06
>>>>>>> 4b64958 (init: algolab-ml（清洗日志 + 特征工程）)
