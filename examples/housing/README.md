# Housing 示例（房价回归）
快速运行：
```bash
algolab-mlrun \
  --csv examples/housing/data/demo_housing.csv \
  --target price \
  --task regression \
  --model xgb \
  --clean-config @examples/housing/configs/clean_housing.yaml \
  --feat-config  @examples/housing/configs/feat_housing.yaml \
  --export --out-dir runs/demo_housing_xgb
```
