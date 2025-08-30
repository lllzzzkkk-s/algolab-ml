# algolab_ml/cli/mlrun.py
#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys, subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

# 工件工具（若包内不可用，降级为本地实现）
try:
    from algolab_ml.utils.artifacts import (
        dump_json, dump_joblib, versions_summary, git_commit_sha, make_run_dir
    )
except Exception:
    import joblib, platform, importlib
    def dump_json(p: Path, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    def dump_joblib(p: Path, obj):
        p.parent.mkdir(parents=True, exist_ok=True); joblib.dump(obj, p)
    def versions_summary():
        pkgs = ["python", "pandas", "numpy", "scikit-learn", "lightgbm", "xgboost"]
        out = {"platform": platform.platform()}
        for name in pkgs:
            try:
                if name == "python": out["python"] = sys.version
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
        d = root / ts; d.mkdir(parents=True, exist_ok=True); return d

from algolab_ml.pipelines import tabular as tabular_pipeline
from algolab_ml.models import model_zoo
from algolab_ml.data.cleaning import (
    basic_clean, fill_na, enforce_schema, drop_constant_cols,
    clip_outliers, bucket_rare_categories, parse_dates
)
from algolab_ml.utils.config import load_json_or_yaml
from algolab_ml.features.builder import FeatureBuilder

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

def _print_list_models(fmt: str, scope: str):
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
        scopes = ["classification","regression"] if scope=="all" else [scope]
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

def main():
    ap = argparse.ArgumentParser(description="Algolab ML Runner（自动/指定任务 + 清洗日志 + CV/搜索/导出）")
    ap.add_argument("--csv", help="训练数据 CSV 路径")
    ap.add_argument("--target", help="目标列名")
    ap.add_argument("--model", default="xgb", help="xgb|lgbm|catboost|rf|gbdt|logreg|ridge|lasso（支持别名）")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--no-preprocess", dest="no_preprocess", action="store_true", help="关闭标准化/One-Hot 等预处理")
    ap.add_argument("--task", choices=["classification","regression","auto"], default="auto", help="覆盖自动识别；默认 auto")
    ap.add_argument("--export", action="store_true", help="保存模型与报告到 --out-dir（未指定则 runs/时间戳）")
    ap.add_argument("--out-dir", default=None, help="保存目录（默认 runs/YYYYmmdd-HHMMSS）")
    ap.add_argument("--params", default=None, help="超参：JSON 或 'k=v,k2=v2'；也支持 @/path/params.json")
    # 清洗 & 特征工程
    ap.add_argument("--clean-config", default=None, help="清洗配置：JSON 文本或 @/path/to/clean.yml(JSON/YAML)")
    ap.add_argument("--feat-config",  default=None, help="特征配置：JSON 文本或 @/path/to/features.yml(JSON/YAML)")
    # 发现性
    ap.add_argument("--list-models", action="store_true", help="列出可用模型与别名")
    ap.add_argument("--format", choices=["json","text"], default="json", help="list 输出格式")
    ap.add_argument("--task-scope", choices=["classification","regression","all"], default="all", help="list 作用域")
    ap.add_argument("--show-params", default=None, help="显示某模型构造函数参数签名后退出")
    # 任务 & 搜索
    ap.add_argument("--cv", type=int, default=None, help="启用交叉验证（折数）")
    ap.add_argument("--scoring", default=None, help="sklearn scoring 名；如 'f1_macro','roc_auc','r2'")
    ap.add_argument("--search", choices=["grid","random","none"], default="none", help="参数搜索方式")
    ap.add_argument("--param-grid", default=None, help="JSON 或 @/path/to/param_grid.json")
    ap.add_argument("--n-iter", type=int, default=20, help="随机搜索迭代次数")
    # 预测导出
    ap.add_argument("--save-preds", action="store_true", help="导出全量数据的预测 out_dir/preds.csv")
    ap.add_argument("--id-col", default=None, help="导出预测时作为 id 的列名")
    args, _ = ap.parse_known_args()

    # 发现类命令
    if args.list_models:
        _print_list_models(args.format, args.task_scope); return
    if args.show_params:
        _print_params(args.show_params); return

    if not args.csv or not args.target:
        raise SystemExit("缺少 --csv / --target。也可以先用 --list-models 或 --show-params。")

    # 读取 & 清洗
    df = pd.read_csv(args.csv)
    clean_cfg = load_json_or_yaml(args.clean_config)

    if clean_cfg:
        # 清洗前探针（考虑 rename）
        probe_col = args.target
        es = (clean_cfg.get("enforce_schema") or {})
        rename = (es.get("rename") or {})
        if probe_col not in df.columns:
            for old, new in rename.items():
                if new == args.target and old in df.columns:
                    probe_col = old; break
        if probe_col in df.columns:
            y_probe = df[probe_col]
            print(f"\n[Task Probe | before clean] col={probe_col}, dtype={y_probe.dtype}, "
                  f"n_unique={y_probe.nunique()}, head={y_probe.head(5).tolist()}")

        df = _apply_cleaning_with_log(df, clean_cfg, target=args.target)

        if args.target in df.columns:
            y_after = df[args.target]
            print(f"[Task Probe | after clean]  col={args.target}, dtype={y_after.dtype}, "
                  f"n_unique={y_after.nunique()}, head={y_after.head(5).tolist()}")
        else:
            print(f"[Task Probe | after clean]  仍找不到目标列 '{args.target}'，请检查清洗配置的 rename/dtypes。")
    # 特征工程
    feat_cfg = load_json_or_yaml(args.feat_config)
    if feat_cfg:
        print("\n------- 开始特征工程（配置式） -------")
        builder = FeatureBuilder(feat_cfg, log_fn=print)
        df = builder.fit_transform(df, target_col=args.target)
        print(f"特征工程新列数：{len(builder.created_)}")

    # 解析参数
    model_params = _parse_params(args.params)
    param_grid = load_json_or_yaml(args.param_grid)
    task_opt = None if args.task == "auto" else args.task

    # 训练
    print("\n======= 开始训练 =======")
    pipe, report = tabular_pipeline.run_df(
        df, target=args.target, model=args.model,
        test_size=args.test_size, random_state=args.random_state,
        preprocess=(not args.no_preprocess),
        model_params=model_params,
        task=task_opt,
        cv=args.cv if args.cv and args.cv > 1 else None,
        scoring=args.scoring,
        search=(None if args.search in (None,"none") else args.search),
        param_grid=param_grid,
        n_iter=args.n_iter
    )
    print("======= 训练完成，评估报告 =======")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 导出
    if args.export:
        out_dir = Path(args.out_dir) if args.out_dir else make_run_dir(Path("runs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_joblib(out_dir / "pipeline.joblib", pipe)
        if clean_cfg: dump_json(out_dir / "clean_config.json", clean_cfg)
        if feat_cfg:
            try:
                feat_conf_to_save = builder.to_dict()
            except Exception:
                feat_conf_to_save = feat_cfg
            dump_json(out_dir / "features_config.json", feat_conf_to_save)
        dump_json(out_dir / "metrics.json", report)
        dump_json(out_dir / "columns.json", {"target": args.target, "all_columns": df.columns.tolist()})
        dump_json(out_dir / "versions.json", versions_summary())
        sha = git_commit_sha()
        if sha: dump_json(out_dir / "git.json", {"commit": sha})

        # 导出预测
        if args.save_preds:
            X_all = df.drop(columns=[args.target])
            y_pred = pipe.predict(X_all)
            score = None
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_all)
                score = proba[:,1] if proba.ndim==2 and proba.shape[1]==2 else None
            out = {}
            if args.id_col and args.id_col in df.columns:
                out[args.id_col] = df[args.id_col].values
            out["pred"] = y_pred
            if score is not None:
                out["score"] = score
            pd.DataFrame(out).to_csv(out_dir / "preds.csv", index=False)

        # 分类图表
        try:
            from algolab_ml.utils.plots import save_roc_pr, save_confusion
            if report.get("task") == "classification":
                X_all = df.drop(columns=[args.target])
                y_all = df[args.target]
                y_pred_all = pipe.predict(X_all)
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(X_all)
                    score = proba[:,1] if proba.ndim==2 and proba.shape[1]==2 else None
                    if score is not None:
                        save_roc_pr(y_all, score, out_dir)
                save_confusion(y_all, y_pred_all, out_dir)
        except Exception as e:
            print(f"[warn] 绘图失败：{e}")

        print(f"📦 训练工件已导出到: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
