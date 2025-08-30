# algolab_ml/cli/mlrun.py
#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys, subprocess
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import pandas as pd

# --- JSON 序列化兜底：numpy/pandas → 纯 Python ---
def _json_default(o):
    import numpy as np
    import pandas as pd
    # NumPy: 标量 & 数组
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    # pandas
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if hasattr(pd, "NA") and o is pd.NA:
        return None
    if isinstance(o, pd.Series):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return {"columns": list(o.columns), "data": o.to_dict(orient="records")}
    return str(o)

def _save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)

# —— 可选工具：若包内 utils 不存在，则用本文件内兜底实现
try:
    from algolab_ml.utils.artifacts import (
        dump_json, dump_joblib, versions_summary, git_commit_sha, make_run_dir
    )
except Exception:  # 兜底
    import joblib, platform, importlib
    def dump_json(p: Path, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    def dump_joblib(p: Path, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, p)
    def versions_summary():
        pkgs = ["python", "pandas", "numpy", "scikit-learn", "lightgbm", "xgboost"]
        out = {"platform": platform.platform()}
        for name in pkgs:
            try:
                if name == "python":
                    out["python"] = sys.version
                else:
                    m = importlib.import_module(name.replace("-", "_"))
                    out[name] = getattr(m, "__version__", "unknown")
            except Exception:
                out[name] = "missing"
        return out
    def git_commit_sha():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None
    def make_run_dir(root: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = root / ts
        d.mkdir(parents=True, exist_ok=True)
        return d

from algolab_ml.pipelines import tabular as tabular_pipeline
from algolab_ml.models import model_zoo
from algolab_ml.data.cleaning import (
    basic_clean, fill_na, enforce_schema, drop_constant_cols,
    clip_outliers, bucket_rare_categories, parse_dates
)
from algolab_ml.utils.config import load_json_or_yaml
from algolab_ml.features.builder import FeatureBuilder

# 可选：绘图（若不存在则自动跳过绘图）
try:
    from algolab_ml.utils.plots import (
        plot_roc_pr_curves,
        plot_feature_importance,
        save_cv_results,
        plot_learning_curve,
    )
except Exception:
    def plot_roc_pr_curves(*args, **kwargs): pass
    def plot_feature_importance(*args, **kwargs): pass
    def save_cv_results(*args, **kwargs): pass
    def plot_learning_curve(*args, **kwargs): pass


# ----------------- 日志工具 -----------------
def _sec(title: str): print(f"\n------- {title} -------")
def _print_df_shape(df: pd.DataFrame, note: str): print(f"{note}：{df.shape[0]} 行 × {df.shape[1]} 列")


def _parse_params(s: str | None) -> dict:
    if not s: return {}
    s = s.strip()
    if s.startswith("@"):
        p = Path(s[1:])
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict): raise ValueError("params 文件必须是 JSON 对象")
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
        if rename:  print(f"重命名列：{rename}")
        if required: print(f"必需列：{required}")
        if dtypes:  print(f"类型约束（目标类型）：{dtypes}")
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
        if target and target in const_cols:
            const_cols = [c for c in const_cols if c != target]
            print(f"已自动排除目标列：{target}")
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
        if cols is None:
            cols = cur.select_dtypes(include=["number"]).columns.tolist()
        if target and target in cols:
            cols = [c for c in cols if c != target]
            print(f"已自动排除目标列：{target}")
        if not cols:
            _sec("异常值截断"); print("没有可处理的数值列，跳过。")
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


# ----------------- 预测/打分（训练后或纯预测模式通用） -----------------
def _run_predict(
    pipe,
    df_pred: pd.DataFrame,
    *,
    proba: bool = False,
    id_cols: list[str] | None = None,
    out_path: str | None = None,
    task_hint: str | None = None,
    expected_input_cols: list[str] | None = None,
    target_col: str | None = None,   # 训练时的 target 名
):
    import numpy as np

    # 1) 分离 id 列
    id_cols = [c for c in (id_cols or []) if c in df_pred.columns]
    id_frame = df_pred[id_cols].copy() if id_cols else None

    # 2) 明确去掉 target 列（防止混入）
    if target_col and target_col in df_pred.columns:
        df_pred = df_pred.drop(columns=[target_col])

    # 3) 计算训练时的输入列（并确保不含 target）
    if expected_input_cols is None:
        expected_input_cols = getattr(pipe, "feature_names_in_", None)
        if expected_input_cols is None:
            expected_input_cols = [c for c in df_pred.columns if c not in id_cols]
    if target_col:
        expected_input_cols = [c for c in expected_input_cols if c != target_col]

    # 4) 对齐列：多退少补，顺序一致
    drop_cols = set(id_cols)
    X_pred = df_pred.drop(columns=list(drop_cols & set(df_pred.columns)), errors="ignore")
    X_pred = X_pred.reindex(columns=list(expected_input_cols), fill_value=np.nan)

    # 5) 预测
    if proba and hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(X_pred)
        out_df = pd.DataFrame(prob, columns=[f"proba_{i}" for i in range(prob.shape[1])])
    else:
        pred = pipe.predict(X_pred)
        out_df = pd.DataFrame({"pred": pred})

    # 6) 拼回 id 列 & 导出
    if id_frame is not None:
        out_df = pd.concat([id_frame.reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"✅ 预测结果已保存到：{Path(out_path).resolve()}")
    else:
        print("（未提供 --predict-out，以下为预测结果预览）")
        print(out_df.head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Algolab ML Runner（自动/指定任务，CV/早停/可视化，带清洗日志 + 推理）")
    # 训练相关
    ap.add_argument("--csv", help="训练数据 CSV 路径")
    ap.add_argument("--target", help="目标列名")
    ap.add_argument("--model", default="xgb", help="xgb|lgbm|catboost|rf|gbdt|logreg|ridge|lasso（支持别名）")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--no-preprocess", dest="no_preprocess", action="store_true", help="关闭标准化/One-Hot 等预处理")
    ap.add_argument("--task", choices=["classification","regression","auto"], default="auto", help="覆盖自动识别；默认 auto")
    # 清洗 & 特征工程
    ap.add_argument("--clean-config", default=None, help="清洗配置：JSON 文本或 @/path/to/clean.yml(JSON/YAML)")
    ap.add_argument("--feat-config",  default=None, help="特征配置：JSON 文本或 @/path/to/features.yml(JSON/YAML)")
    # CV / 搜索
    ap.add_argument("--cv", type=int, default=0, help="交叉验证折数（0/1 表示不做 CV）")
    ap.add_argument("--search", choices=["grid","random"], default="grid", help="超参搜索方式")
    ap.add_argument("--n-iter", type=int, default=20, help="RandomizedSearch 的迭代次数")
    ap.add_argument("--param-grid", default=None, help="指定搜索空间（JSON 或 @/path/file.json），不填用内置默认")
    ap.add_argument("--scoring", default=None, help="scoring 名称（如 roc_auc / accuracy / f1_macro / r2 / neg_root_mean_squared_error 等）")
    # 早停
    ap.add_argument("--early-stopping", action="store_true", help="启用早停（LightGBM/XGBoost 生效）")
    ap.add_argument("--val-size", type=float, default=0.15, help="早停的验证集占训练集比例")
    ap.add_argument("--es-rounds", type=int, default=50, help="没有提升的容忍迭代数")
    ap.add_argument("--eval-metric", default=None, help="eval_metric（如 auc / logloss / rmse 等）")
    # 导出
    ap.add_argument("--export", action="store_true", help="保存模型与报告到 --out-dir（未指定则 runs/时间戳）")
    ap.add_argument("--out-dir", default=None, help="保存目录（默认 runs/YYYYmmdd-HHMMSS）")
    ap.add_argument("--params", default=None, help="模型超参数，JSON 或 'k=v,k2=v2' 或 @/path/params.json")
    # 预测/打分
    ap.add_argument("--model-path", default=None, help="已有模型管道路径（pipeline.joblib），仅预测模式必填或配合 --run-dir")
    ap.add_argument("--run-dir", default=None, help="训练导出目录（包含 pipeline.joblib / columns.json / features_config.json / clean_config.json）")
    ap.add_argument("--predict-csv", default=None, help="需要打分的 CSV（可配合 --model-path 或训练后直接打分）")
    ap.add_argument("--predict-out", default=None, help="预测结果输出 CSV（默认打印前 10 行）")
    ap.add_argument("--proba", action="store_true", help="分类任务输出正类概率 proba")
    ap.add_argument("--id-cols", default=None, help="逗号分隔的 id 列名，原样拷贝到预测结果")
    # 发现性
    ap.add_argument("--list-models", action="store_true", help="列出可用模型与别名")
    ap.add_argument("--format", choices=["json","text"], default="json", help="list 输出格式")
    ap.add_argument("--task-scope", choices=["classification","regression","all"], default="all", help="list 作用域")
    ap.add_argument("--show-params", default=None, help="显示某模型构造函数参数签名后退出")
    args, _ = ap.parse_known_args()

    # 发现类命令
    if args.list_models:
        _print_list_models(args.format, args.task_scope); return
    if args.show_params:
        _print_params(args.show_params); return

    # ----------------- 纯“预测模式”（不训练） -----------------
    if args.predict_csv and not args.csv:
        if not args.model_path and not args.run_dir:
            raise SystemExit("仅预测模式需要提供 --model-path 或 --run-dir")

        # 解析路径
        run_dir = Path(args.run_dir) if args.run_dir else None
        model_path = Path(args.model_path) if args.model_path else None
        if run_dir and not model_path:
            mp1 = run_dir / "pipeline.joblib"
            if mp1.exists():
                model_path = mp1
        if not model_path or not model_path.exists():
            raise SystemExit(f"找不到模型：{model_path or '(未提供)'}")

        # 读取待预测
        df_pred = pd.read_csv(args.predict_csv)

        # 尝试加载训练时元数据
        expected_cols = None
        target_col = getattr(args, "target", None)
        if run_dir:
            cols_file = run_dir / "columns.json"
            if cols_file.exists():
                cols_json = json.loads(cols_file.read_text(encoding="utf-8"))
                all_cols = cols_json.get("all_columns") or []
                tgt = cols_json.get("target")
                if all_cols:
                    expected_cols = [c for c in all_cols if c != tgt]
                if tgt:
                    target_col = tgt

        # 清洗/特征工程（transform）
        clean_cfg = None
        feat_cfg = None
        builder = None
        if run_dir:
            cc = run_dir / "clean_config.json"
            fc = run_dir / "features_config.json"
            if cc.exists():
                clean_cfg = json.loads(cc.read_text(encoding="utf-8"))
                # 预测阶段：尽量不动列结构 —— 跳过 drop_constant
                if isinstance(clean_cfg, dict) and "drop_constant" in clean_cfg:
                    print("⚠️  预测阶段为保持列一致，已跳过 drop_constant（删除常量列）")
                    clean_cfg = deepcopy(clean_cfg)
                    clean_cfg.pop("drop_constant", None)
                df_pred = _apply_cleaning_with_log(df_pred, clean_cfg, target=target_col)
            if fc.exists():
                feat_cfg = json.loads(fc.read_text(encoding="utf-8"))
                if isinstance(feat_cfg, dict) and feat_cfg:
                    print("\n------- 预测阶段：按训练时配置做特征工程（transform） -------")
                    builder = FeatureBuilder(feat_cfg, log_fn=print)
                    df_pred = builder.transform(df_pred)

        # 加载 pipeline 并预测
        from joblib import load as _load
        pipe = _load(model_path)
        id_cols = [s.strip() for s in args.id_cols.split(",")] if args.id_cols else None
        out_path = Path(args.predict_out) if args.predict_out else None

        _run_predict(
            pipe,
            df_pred,
            proba=bool(getattr(args, "proba", False)),
            id_cols=id_cols,
            out_path=str(out_path) if out_path else None,
            expected_input_cols=expected_cols,
            target_col=target_col,
        )
        return

    # ----------------- 训练路径 -----------------
    if not args.csv or not args.target:
        raise SystemExit("训练模式需要 --csv / --target。也可以使用“仅预测模式”。")

    # 读取 CSV
    df = pd.read_csv(args.csv)
    clean_cfg = load_json_or_yaml(args.clean_config)

    # 清洗（带日志）
    if clean_cfg:
        df = _apply_cleaning_with_log(df, clean_cfg, target=args.target)

    # 特征工程
    feat_cfg = load_json_or_yaml(args.feat_config)
    builder = None
    if isinstance(feat_cfg, dict) and feat_cfg:
        print("\n------- 开始特征工程（配置式） -------")
        builder = FeatureBuilder(feat_cfg, log_fn=print)
        df = builder.fit_transform(df, target_col=args.target)
        print(f"特征工程新列数：{len(builder.created_)}")
    else:
        print("\n------- 特征工程完成 -------")
        print("本次新增特征列总数：0")
        print("特征工程新列数：0")

    # 训练
    model_params = _parse_params(getattr(args, "params", None))
    print("\n======= 开始训练 =======")
    pipe, report = tabular_pipeline.run_df(
        df, target=args.target, model=args.model,
        test_size=args.test_size, random_state=args.random_state,
        preprocess=(not args.no_preprocess),
        model_params=model_params,
        task=(None if args.task == "auto" else args.task),
        cv=args.cv, search=args.search, n_iter=args.n_iter,
        scoring=args.scoring, param_grid=args.param_grid,
        early_stopping=args.early_stopping, val_size=args.val_size,
        es_rounds=args.es_rounds, eval_metric=args.eval_metric,
    )
    print("======= 训练完成，评估报告 =======")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))

    # 导出
    if args.export:
        out_dir = Path(args.out_dir) if args.out_dir else make_run_dir(Path("runs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_joblib(out_dir / "pipeline.joblib", pipe)
        report_to_save = {k: v for k, v in report.items() if not str(k).startswith("_")}
        _save_json(out_dir / "metrics.json", report_to_save)
        _save_json(out_dir / "columns.json", {"target": args.target, "all_columns": df.columns.tolist()})
        _save_json(out_dir / "versions.json", versions_summary())
        if clean_cfg: _save_json(out_dir / "clean_config.json", clean_cfg)
        if builder is not None:
            try:
                _save_json(out_dir / "features_config.json", builder.to_dict())
            except Exception:
                if feat_cfg: _save_json(out_dir / "features_config.json", feat_cfg)

        # 曲线与可视化
        try:
            if report.get("task") == "classification" and "roc_auc" in report and "_y_true" in report and "_y_prob" in report:
                plot_roc_pr_curves(report["_y_true"], report["_y_prob"], out_dir)
                print("✅ 已保存 ROC/PR 曲线")
        except Exception:
            pass
        try:
            feature_names = getattr(pipe, "feature_names_in_", None)
            plot_feature_importance(pipe, feature_names, out_dir, top_n=30)
            print("✅ 已保存特征重要性")
        except Exception:
            pass
        try:
            # CV 结果（在做过 Grid/RandomSearch 时有效）
            save_cv_results(pipe, out_dir)
        except Exception:
            pass
        try:
            # 学习曲线（早停训练）
            ev = report.get("evals_result")
            if ev:
                plot_learning_curve(ev, out_dir, metric_hint=args.eval_metric, task=report.get("task"))
                print("✅ 已保存学习曲线")
        except Exception:
            pass

        print(f"📦 训练工件已导出到: {out_dir.resolve()}")

    # ------- 训练后立即对新 CSV 预测 -------
    pred_csv = getattr(args, "predict_csv", None)
    pred_out = getattr(args, "predict_out", None)
    proba    = bool(getattr(args, "proba", False))
    id_cols_arg = getattr(args, "id_cols", None)

    if pred_csv:
        p = Path(pred_csv)
        if not p.exists():
            print(f"⚠️  预测文件不存在：{pred_csv}，已跳过预测。")
        else:
            print("\n------- 训练后立即对新 CSV 预测 -------")
            import numpy as np

            df_pred_raw = pd.read_csv(p)

            # 1)（可选）执行与训练一致的清洗（尽量保持列一致）
            if clean_cfg:
                clean_cfg_pred = deepcopy(clean_cfg)
                if isinstance(clean_cfg_pred, dict) and "drop_constant" in clean_cfg_pred:
                    print("⚠️  预测阶段为保持列一致，已跳过 drop_constant（删除常量列）")
                    clean_cfg_pred.pop("drop_constant", None)
                df_pred = _apply_cleaning_with_log(df_pred_raw, clean_cfg_pred, target=args.target)
            else:
                df_pred = df_pred_raw.copy()

            # 2)（可选）执行与训练一致的特征工程（builder 已在训练阶段 fit 过）
            if builder is not None:
                try:
                    df_pred = builder.transform(df_pred)
                except Exception as e:
                    print(f"⚠️ 特征工程在预测阶段 transform 失败：{e}。将直接用原始列进行预测。")

            # 3) 取出 id 列（若用户指定），并从特征中排除
            id_cols = []
            if id_cols_arg:
                id_cols = [c.strip() for c in str(id_cols_arg).split(",") if c.strip() and c.strip() in df_pred.columns]
            id_frame = df_pred[id_cols].copy() if id_cols else None

            # 4) 按“训练后的 df 列”作为金标准对齐（排除 target）
            expected_cols = [c for c in df.columns if c != args.target]
            drop_cols = [args.target] if args.target in df_pred.columns else []
            drop_cols += [c for c in id_cols if c in df_pred.columns]
            X_pred = df_pred.drop(columns=drop_cols, errors="ignore")
            X_pred = X_pred.reindex(columns=expected_cols, fill_value=np.nan)

            # 5) 预测
            if proba and hasattr(pipe, "predict_proba"):
                prob = pipe.predict_proba(X_pred)
                proba_cols = [f"proba_{i}" for i in range(prob.shape[1])]
                out_df = pd.DataFrame(prob, columns=proba_cols)
            else:
                pred = pipe.predict(X_pred)
                out_df = pd.DataFrame({"pred": pred})

            # 拼回 id 列并导出
            if id_frame is not None:
                out_df = pd.concat([id_frame.reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)

            if pred_out:
                Path(pred_out).parent.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(pred_out, index=False)
                print(f"✅ 预测结果已保存到：{Path(pred_out).resolve()}")
            else:
                print("（未提供 --predict-out，以下为预测结果预览）")
                print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
