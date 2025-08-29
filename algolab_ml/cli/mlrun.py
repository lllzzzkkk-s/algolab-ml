#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

from algolab_ml.pipelines import tabular as tabular_pipeline
from algolab_ml.models import model_zoo
from algolab_ml.data.cleaning import (
    basic_clean, fill_na, enforce_schema, drop_constant_cols,
    clip_outliers, bucket_rare_categories, parse_dates
)
from algolab_ml.utils.config import load_json_or_yaml
from algolab_ml.features.builder import FeatureBuilder

# ----------------- 工具函数 -----------------
def _sec(title: str):
    print(f"\n------- {title} -------")

def _print_df_shape(df: pd.DataFrame, note: str):
    print(f"{note}：{df.shape[0]} 行 × {df.shape[1]} 列")

def _parse_params(s: str | None) -> dict:
    if not s: return {}
    s = s.strip()
    if s.startswith("@"):
        p = Path(s[1:])
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("params file must be一个 JSON 对象")
        return data
    try:
        data = json.loads(s)
        if isinstance(data, dict): return data
    except Exception:
        pass
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part: continue
        if "=" not in part: raise ValueError(f"Bad param '{part}', expected key=value")
        k, v = part.split("=", 1); k, v = k.strip(), v.strip()
        if v.lower() in ("true","false"): out[k] = (v.lower()=="true"); continue
        try: out[k] = int(v); continue
        except: pass
        try: out[k] = float(v); continue
        except: pass
        out[k] = v
    return out

def _print_list_models(fmt: str, task: str):
    data = {
        "classification": {
            "canonical": model_zoo.available_models("classification"),
            "aliases": model_zoo.aliases_for_task("classification"),
        },
        "regression": {
            "canonical": model_zoo.available_models("regression"),
            "aliases": model_zoo.aliases_for_task("regression"),
        },
    }
    if fmt == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        scopes = ["classification","regression"] if task=="all" else [task]
        for sc in scopes:
            print(f"[{sc}]")
            print("  models: " + ", ".join(sorted(data[sc]["canonical"])))
            print("  aliases:")
            for k, v in sorted(data[sc]["aliases"].items()):
                print(f"    {k} -> {v}")

def _print_params(name: str):
    out = {}
    for task in ("classification","regression"):
        try:
            out[task] = model_zoo.model_signature(name, task)
        except KeyError:
            pass
    if not out:
        avail = {
            "classification": model_zoo.available_models("classification"),
            "regression": model_zoo.available_models("regression"),
        }
        raise SystemExit(f"Unknown model alias '{name}'. Available: {json.dumps(avail, ensure_ascii=False)}")
    print(json.dumps(out, ensure_ascii=False, indent=2))

# ----------------- 带日志的清洗执行 -----------------
def _apply_cleaning_with_log(df: pd.DataFrame, cfg: dict, target: str | None = None) -> pd.DataFrame:
    cur = df.copy()
    _sec("开始数据清洗（配置式）")
    _print_df_shape(cur, "原始数据维度")

    # 1) enforce_schema
    es = cfg.get("enforce_schema")
    if es:
        _sec("Schema 规范化")
        rename = es.get("rename") or {}
        required = es.get("required") or []
        dtypes = es.get("dtypes") or {}
        if rename:
            print(f"重命名列：{rename}")
        if required:
            print(f"必需列：{required}")
        if dtypes:
            print(f"类型约束（目标类型）：{dtypes}")
        cur = enforce_schema(cur, dtypes=dtypes, rename=rename, required=required)
        _print_df_shape(cur, "规范化后维度")

    # 2) basic_clean
    bc = cfg.get("basic_clean")
    if bc:
        drop_cols = bc.get("drop_cols") if isinstance(bc, dict) else None
        dup_cnt = cur.duplicated().sum()
        before_cols = set(cur.columns)
        cur2 = basic_clean(cur, drop_cols=drop_cols)
        dropped_cols = list(before_cols - set(cur2.columns))
        print(f"去重行数：{dup_cnt}")
        if drop_cols:
            kept = [c for c in drop_cols if c in before_cols]
            print(f"删除指定列：{kept or []}")
        if dropped_cols:
            print(f"（含重复名/无效导致的）实际被删除列：{dropped_cols}")
        cur = cur2
        _print_df_shape(cur, "basic_clean 后维度")

    # 3) fill_na
    if cfg.get("fill_na"):
        na = cfg["fill_na"]
        num = na.get("num","median")
        cat = na.get("cat","most_frequent")
        num_cols = cur.select_dtypes(include=["number"]).columns
        cat_cols = [c for c in cur.columns if c not in num_cols]
        n_na_num_before = int(cur[num_cols].isna().sum().sum()) if len(num_cols)>0 else 0
        n_na_cat_before = int(cur[cat_cols].isna().sum().sum()) if len(cat_cols)>0 else 0
        cur2 = fill_na(cur, num_strategy=num, cat_strategy=cat)
        n_na_num_after = int(cur2[num_cols].isna().sum().sum()) if len(num_cols)>0 else 0
        n_na_cat_after = int(cur2[cat_cols].isna().sum().sum()) if len(cat_cols)>0 else 0
        _sec("缺失值填充")
        print(f"数值列填充策略：{num}，缺失 {n_na_num_before} -> {n_na_num_after}")
        print(f"类别列填充策略：{cat}，缺失 {n_na_cat_before} -> {n_na_cat_after}")
        cur = cur2

    # 4) drop_constant
    if cfg.get("drop_constant"):
        th = cfg["drop_constant"].get("threshold_unique", 1)
        const_cols = [c for c in cur.columns if cur[c].nunique(dropna=True) <= th]
        cur = drop_constant_cols(cur, threshold_unique=th)
        _sec("删除常量列")
        print(f"阈值：唯一值 ≤ {th}，删除列：{const_cols}")

    # 5) clip_outliers
    if cfg.get("clip_outliers"):
        co = cfg["clip_outliers"]
        method = co.get("method","iqr")
        zt = co.get("z_thresh",3.0)
        ik = co.get("iqr_k",1.5)
        cols = co.get("cols")
        # 自动选择数值列
        if cols is None:
            cols = cur.select_dtypes(include=["number"]).columns.tolist()
        # 排除目标列（避免对 label/目标做截断）
        if target and target in cols:
            cols = [c for c in cols if c != target]
            print(f"已自动排除目标列：{target}")

        # 如果剩余可处理列为空，跳过
        if not cols:
            _sec("异常值截断")
            print("没有可处理的数值列，跳过。")
        else:
            before = cur[cols].copy()
            cur = clip_outliers(cur, cols=cols, method=method, z_thresh=zt, iqr_k=ik)
            changed = {c: int((cur[c] != before[c]).sum()) for c in cols if c in cur.columns}
            total_changed = int(sum(changed.values()))
            _sec("异常值截断")
            print(f"方式：{method}（z={zt} / iqr_k={ik}），处理列数：{len(cols)}，被截断的单元格总数：{total_changed}")
            if total_changed:
                top = sorted(changed.items(), key=lambda x: x[1], reverse=True)[:8]
                print(f"变化最多的列（前 8）：{top}")
    # 6) bucket_rare
    if cfg.get("bucket_rare"):
        br = cfg["bucket_rare"]
        cols = br.get("cols", [])
        minf = br.get("min_freq", 10)
        other = br.get("other_label","_OTHER")
        counts = {}
        for c in cols:
            if c in cur.columns:
                vc = cur[c].astype(str).value_counts(dropna=False)
                counts[c] = int((vc < minf).sum())
        cur = bucket_rare_categories(cur, cols=cols, min_freq=minf, other_label=other)
        _sec("低频类别合并")
        print(f"列：{cols}，频次阈值：{minf}，合并标签：{other}")
        if counts:
            print(f"各列低频类别个数：{counts}")

    # 7) parse_dates
    if cfg.get("parse_dates"):
        mapping = cfg["parse_dates"]
        cur = parse_dates(cur, mapping)
        _sec("日期解析")
        info = {}
        for col in mapping:
            if col in cur.columns:
                ok = int(cur[col].notna().sum())
                na = int(cur[col].isna().sum())
                info[col] = {"可解析": ok, "NaT": na}
        print(f"解析列：{mapping}，结果：{info}")

    _sec("清洗完成")
    _print_df_shape(cur, "最终数据维度")
    return cur

# ----------------- 主程序 -----------------
def main():
    ap = argparse.ArgumentParser(description="Algolab ML Runner（带清洗日志）")
    ap.add_argument("--csv", help="训练数据 CSV 路径")
    ap.add_argument("--target", help="目标列名")
    ap.add_argument("--model", default="xgb", help="xgb|lgbm|catboost|rf|gbdt|logreg|ridge|lasso（支持别名）")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--no-preprocess", dest="no_preprocess", action="store_true", help="关闭标准化/One-Hot 等预处理")
    ap.add_argument("--export", action="store_true", help="保存模型与报告到 --out-dir（未指定则 runs/时间戳）")
    ap.add_argument("--out-dir", default=None, help="保存目录（默认 runs/YYYYmmdd-HHMMSS）")
    ap.add_argument("--params", default=None, help="超参：JSON 或 'k=v,k2=v2'；也支持 @/path/params.json")
    # 清洗
    ap.add_argument("--clean-config", default=None, help="清洗配置：JSON 文本或 @/path/to/clean.yml(JSON/YAML)")
    ap.add_argument("--clean-basic", action="store_true", help="执行 basic_clean（去重、列名清理等）")
    ap.add_argument("--fill-na", default=None, help="缺失值策略：'num=median,cat=most_frequent' 或 JSON")
    ap.add_argument("--feat-config", default=None, help="特征工程配置：JSON 文本或 @/path/to/features.yml(JSON/YAML)")
    # 发现性
    ap.add_argument("--list-models", action="store_true", help="列出可用模型与别名")
    ap.add_argument("--format", choices=["json","text"], default="json", help="list 输出格式")
    ap.add_argument("--task", choices=["classification","regression","all"], default="all", help="list 作用域")
    ap.add_argument("--show-params", default=None, help="显示某模型构造函数参数签名后退出")
    args, unknown = ap.parse_known_args()

    # 发现型命令
    if args.list_models:
        _print_list_models(args.format, args.task); return
    if args.show_params:
        _print_params(args.show_params); return

    if not args.csv or not args.target:
        raise SystemExit("缺少 --csv / --target。也可以先用 --list-models 或 --show-params。")

    # 读取 & 清洗
    df = pd.read_csv(args.csv)
    cfg = load_json_or_yaml(args.clean_config)

    if cfg:
        df = _apply_cleaning_with_log(df, cfg, target=args.target)
    else:
        if args.clean_basic:
            _sec("basic_clean")
            _print_df_shape(df, "清洗前")
            dup_cnt = df.duplicated().sum()
            df = basic_clean(df)
            print(f"去重行数：{dup_cnt}")
            _print_df_shape(df, "清洗后")
        if args.fill_na:
            _sec("缺失值填充")
            try:
                if args.fill_na.strip().startswith("{"):
                    na = json.loads(args.fill_na)
                    num = na.get("num","median"); cat = na.get("cat","most_frequent")
                else:
                    parts = dict(p.split("=") for p in args.fill_na.split(","))
                    num = parts.get("num","median"); cat = parts.get("cat","most_frequent")
            except Exception:
                num, cat = "median", "most_frequent"
            num_cols = df.select_dtypes(include=["number"]).columns
            cat_cols = [c for c in df.columns if c not in num_cols]
            n_na_num_before = int(df[num_cols].isna().sum().sum()) if len(num_cols)>0 else 0
            n_na_cat_before = int(df[cat_cols].isna().sum().sum()) if len(cat_cols)>0 else 0
            df = fill_na(df, num_strategy=num, cat_strategy=cat)
            n_na_num_after = int(df[num_cols].isna().sum().sum()) if len(num_cols)>0 else 0
            n_na_cat_after = int(df[cat_cols].isna().sum().sum()) if len(cat_cols)>0 else 0
            print(f"数值缺失：{n_na_num_before} -> {n_na_cat_after}")
            print(f"类别缺失：{n_na_cat_before} -> {n_na_cat_after}")
            _print_df_shape(df, "填充后")
    
    # 清洗完成后
    feat_cfg = load_json_or_yaml(args.feat_config)
    if feat_cfg:
        print("\n------- 开始特征工程（配置式） -------")
        builder = FeatureBuilder(feat_cfg, log_fn=print)
        df = builder.fit_transform(df, target_col=args.target)
        print(f"特征工程新列数：{len(builder.created_)}")

    # 训练
    model_params = _parse_params(args.params)
    print("\n======= 开始训练 =======")
    pipe, report = tabular_pipeline.run_df(
        df, target=args.target, model=args.model,
        test_size=args.test_size, random_state=args.random_state,
        preprocess=(not args.no_preprocess), **model_params
    )
    print("======= 训练完成，评估报告 =======")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 导出
    if args.export:
        out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
        (out_dir / "features_config.json").write_text(builder.to_json(), encoding="utf-8")
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_dir / "model.joblib")
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📦 已保存: {out_dir/'model.joblib'} / {out_dir/'report.json'}", file=sys.stderr)

if __name__ == "__main__":
    main()
