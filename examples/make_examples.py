# scripts/make_examples.py
from pathlib import Path
import numpy as np, pandas as pd
import datetime as dt
import textwrap

BASE = Path("examples")
CHURN_D = BASE/"churn"/"data"; CHURN_C = BASE/"churn"/"configs"
HOUSE_D = BASE/"housing"/"data"; HOUSE_C = BASE/"housing"/"configs"
TS_D = BASE/"timeseries"/"data"
for p in [CHURN_D, CHURN_C, HOUSE_D, HOUSE_C, TS_D]: p.mkdir(parents=True, exist_ok=True)

# ---------------- 1) Churn（分类） ----------------
def make_churn(n_train=500, n_pred=120, seed=42):
    rng = np.random.default_rng(seed)
    cities = ["Beijing","Shanghai","Shenzhen","Hangzhou","Chengdu","Wuhan","Nanjing","Changsha","Xiamen","Qingdao"]
    plans = ["basic","pro","vip"]
    comments = [
        "great service","bad network","pricing high","helpful staff",
        "app crash","no issues","slow loading","coupon issue"
    ]
    start = dt.date(2020,1,1)
    rows=[]
    for i in range(n_train+n_pred):
        plan = rng.choice(plans, p=[0.5,0.35,0.15])
        fee = {"basic":39,"pro":79,"vip":129}[plan] + rng.normal(0,3)
        tenure = rng.integers(1, 36)
        last_login = rng.integers(0, 90)
        clicks = max(0, int(rng.normal(20, 12)))
        tickets = rng.poisson(1.2 if plan!="vip" else 0.6)
        coupon = rng.choice([0,1], p=[0.7,0.3])
        city = rng.choice(cities, p=np.array([5,6,5,3,4,3,3,2,2,2])/35)
        join_date = start + dt.timedelta(days=int(rng.integers(0, 365*2)))
        comment = rng.choice(comments)
        # label 概率
        score = -1.0 + 0.8*(fee < 50) + 1.1*(tickets > 3) + 0.9*(last_login > 30) \
                + 0.6*(clicks < 10) - 0.6*(coupon==1) - 0.3*(plan=="vip") - 0.2*(tenure>12) \
                + rng.normal(0,0.6)
        p = 1/(1+np.exp(-score))
        label = int(rng.uniform() < p)
        rows.append({
            "id": 100000+i, "city": city, "plan": plan, "join_date": str(join_date),
            "tenure_months": int(tenure), "monthly_fee": round(float(fee),2),
            "last_login_days": int(last_login), "clicks_30d": int(clicks),
            "support_tickets": int(tickets), "coupon_used": int(coupon),
            "comment": comment, "label": label
        })
    df = pd.DataFrame(rows)
    # 缺失
    mask = rng.uniform(size=len(df))<0.10; df.loc[mask, "comment"] = np.nan
    mask = rng.uniform(size=len(df))<0.03; df.loc[mask, "last_login_days"] = np.nan
    # 划分
    df_train = df.iloc[:n_train].copy()
    df_pred  = df.iloc[n_train:].drop(columns=["label"]).copy()
    df_train.to_csv(CHURN_D/"demo_marketing_churn.csv", index=False)
    df_pred.to_csv(CHURN_D/"demo_marketing_churn_new.csv", index=False)

    # 清洗/特征 YAML
    (CHURN_C/"clean_churn.yaml").write_text(textwrap.dedent("""
    enforce_schema:
      required: ["id"]
    basic_clean: {}
    fill_na:
      num: median
      cat: most_frequent
    drop_constant:
      threshold_unique: 1
    clip_outliers:
      method: iqr
      z_thresh: 3.0
      iqr_k: 1.5
    bucket_rare:
      cols: ["city","plan"]
      min_freq: 20
      other_label: "_OTHER"
    parse_dates:
      join_date: auto
    """).strip()+"\n", encoding="utf-8")

    (CHURN_C/"feat_churn.yaml").write_text(textwrap.dedent("""
    polynomial:
      degree: 2
      cols: ["tenure_months","last_login_days"]
    interactions:
      pairs:
        - ["tenure_months","monthly_fee"]
        - ["clicks_30d","support_tickets"]
    binning:
      method: quantile
      bins: 5
      cols: ["monthly_fee"]
    date_expand:
      join_date: ["year","month"]
    freq_encoding:
      cols: ["city","plan"]
    text_features:
      cols: ["comment"]
      metrics: ["length","num_alpha","num_digit"]
    """).strip()+"\n", encoding="utf-8")

    (BASE/"churn"/"README.md").write_text(textwrap.dedent("""
    # Churn 示例（用户是否流失）
    关键字段：plan、monthly_fee、last_login_days、support_tickets、coupon_used、label(目标)
    快速运行：
    ```bash
    algolab-mlrun \\
      --csv examples/churn/data/demo_marketing_churn.csv \\
      --target label \\
      --task classification \\
      --model lgbm \\
      --clean-config @examples/churn/configs/clean_churn.yaml \\
      --feat-config  @examples/churn/configs/feat_churn.yaml \\
      --early-stopping --val-size 0.15 --es-rounds 80 --eval-metric auc \\
      --export --out-dir runs/demo_churn_lgbm \\
      --predict-csv examples/churn/data/demo_marketing_churn_new.csv \\
      --predict-out runs/demo_churn_lgbm/preds.csv \\
      --proba --id-cols id
    ```
    """).strip()+"\n", encoding="utf-8")

# ---------------- 2) Housing（回归） ----------------
def make_housing(m=450, seed=7):
    rng = np.random.default_rng(seed)
    cities = ["Hangzhou","Suzhou","Nanjing","Wuxi","Ningbo"]
    adj = {"Hangzhou":1.15,"Suzhou":1.10,"Nanjing":1.05,"Wuxi":0.95,"Ningbo":1.00}
    rows=[]
    for i in range(m):
        area = rng.uniform(50, 180)
        rooms = rng.integers(1,6)
        age = rng.uniform(0,30)
        dist = abs(rng.normal(8,5))
        city = rng.choice(cities)
        garage = rng.choice([0,1], p=[0.6,0.4])
        base = 12000*area + 50000*rooms - 3000*age - 15000*dist + 80000*garage
        price = base*adj[city] + rng.normal(0, 5e4)
        rows.append({
            "house_id": 200000+i, "city": city, "area": round(float(area),2),
            "rooms": int(rooms), "age_years": round(float(age),1),
            "distance_km": round(float(dist),2), "has_garage": int(garage),
            "price": round(float(max(price, 8e4)),2)
        })
    df = pd.DataFrame(rows)
    mask = rng.uniform(size=len(df))<0.02; df.loc[mask, "distance_km"] = np.nan
    df.to_csv(HOUSE_D/"demo_housing.csv", index=False)

    (HOUSE_C/"clean_housing.yaml").write_text(textwrap.dedent("""
    basic_clean: {}
    fill_na:
      num: median
      cat: most_frequent
    clip_outliers:
      method: iqr
      iqr_k: 1.5
    bucket_rare:
      cols: ["city"]
      min_freq: 15
      other_label: "_OTHER"
    """).strip()+"\n", encoding="utf-8")

    (HOUSE_C/"feat_housing.yaml").write_text(textwrap.dedent("""
    interactions:
      pairs:
        - ["area","rooms"]
        - ["distance_km","age_years"]
    binning:
      method: quantile
      bins: 5
      cols: ["area"]
    freq_encoding:
      cols: ["city"]
    """).strip()+"\n", encoding="utf-8")

    (BASE/"housing"/"README.md").write_text(textwrap.dedent("""
    # Housing 示例（房价回归）
    快速运行：
    ```bash
    algolab-mlrun \\
      --csv examples/housing/data/demo_housing.csv \\
      --target price \\
      --task regression \\
      --model xgb \\
      --clean-config @examples/housing/configs/clean_housing.yaml \\
      --feat-config  @examples/housing/configs/feat_housing.yaml \\
      --export --out-dir runs/demo_housing_xgb
    ```
    """).strip()+"\n", encoding="utf-8")

# ---------------- 3) Timeseries（日销量） ----------------
def make_timeseries(days=240, seed=9):
    rng = np.random.default_rng(seed)
    start = dt.date(2023,1,1)
    rows=[]
    for d in range(days):
        date = start + dt.timedelta(days=d)
        dow = date.weekday()
        season = 1.0 + 0.2*np.sin(2*np.pi*d/30)
        weekend = 1.25 if dow>=5 else 1.0
        promo = rng.choice([0,1], p=[0.85,0.15])
        base = 80*season*weekend*(1.4 if promo else 1.0)
        sales = int(max(rng.normal(base, 10),0))
        rows.append({"date": str(date), "store_id": "S01", "item_id":"I001", "promo": promo, "sales": sales})
    pd.DataFrame(rows).to_csv(TS_D/"demo_retail_timeseries.csv", index=False)

    (BASE/"timeseries"/"README.md").write_text(textwrap.dedent("""
    # Timeseries 示例（日销量）
    仅提供数据，后续将添加时间序列特征与时间序列 CV 示例。
    查看：
    ```bash
    head examples/timeseries/data/demo_retail_timeseries.csv
    ```
    """).strip()+"\n", encoding="utf-8")

if __name__ == "__main__":
    make_churn()
    make_housing()
    make_timeseries()
    print("✅ examples 生成完成")
