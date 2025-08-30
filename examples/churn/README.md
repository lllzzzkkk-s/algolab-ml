# Churn 示例（用户是否流失）
关键字段：plan、monthly_fee、last_login_days、support_tickets、coupon_used、label(目标)
快速运行：
```bash
algolab-mlrun \
  --csv examples/churn/data/demo_marketing_churn.csv \
  --target label \
  --task classification \
  --model lgbm \
  --clean-config @examples/churn/configs/clean_churn.yaml \
  --feat-config  @examples/churn/configs/feat_churn.yaml \
  --early-stopping --val-size 0.15 --es-rounds 80 --eval-metric auc \
  --export --out-dir runs/demo_churn_lgbm \
  --predict-csv examples/churn/data/demo_marketing_churn_new.csv \
  --predict-out runs/demo_churn_lgbm/preds.csv \
  --proba --id-cols id
```
