# algolab_ml/features/builder.py
from __future__ import annotations
import re
import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd

def _slug(s: Any) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_.-]", "_", s)
    return s

class FeatureBuilder:
    """
    统一的特征构建器（最小可用实现）：
    - 支持 target_mean（目标均值编码，OOF 防泄漏 + 推理映射）
      * 回归：单列编码
      * 二分类：单列编码（P(y=正类|x) 的平滑均值）
      * 多分类：多列编码（每个类一列：P(y=c|x) 的平滑均值）

    cfg 结构（示例）：
    {
      "target_mean": {
        "cols": ["city","brand"],
        "alpha": 10,
        "min_samples": 2,
        "n_splits": 5,
        "random_state": 42,
        "suffix": "__tmean"
      }
    }

    持久化：
    - to_dict() 会保存 {"config": cfg, "created": [...], "state": {...}}
    - __init__ 支持传入 {"config":..., "state":...} 或仅 cfg
    - 这样纯预测 (--run-dir) 阶段可直接 transform 使用训练期映射
    """
    def __init__(self, cfg_or_bundle: Dict | None = None, log_fn=print):
        cfg_or_bundle = cfg_or_bundle or {}
        if "config" in cfg_or_bundle or "state" in cfg_or_bundle:
            self.cfg: Dict = cfg_or_bundle.get("config", {})
            state = cfg_or_bundle.get("state", {})
        else:
            self.cfg = cfg_or_bundle
            state = {}

        self.log = log_fn
        self.created_: List[str] = []
        # state_ 结构：
        # {
        #   "tmean": {
        #       "_meta": {...},
        #       "<col>": {
        #           "mode": "regression"|"binary"|"multiclass",
        #           "mapping": Series 或 DataFrame（index=类别；列=类标签（多分类）或单列）或其 dict 形式,
        #           "classes": [仅多分类，类标签列表],
        #           "global": float 或 {cls: float},
        #           "alpha": int
        #       },
        #       ...
        #   }
        # }
        self.state_: Dict = {"tmean": {}}
        # 恢复 state（兼容 dict 形式）
        tmean_saved = state.get("tmean", {})
        if tmean_saved:
            meta = tmean_saved.get("_meta", {})
            self.state_["tmean"]["_meta"] = meta
            for k, pack in tmean_saved.items():
                if k == "_meta":
                    continue
                mode = pack.get("mode", "regression")
                if mode == "multiclass":
                    # mapping: dict[category] -> dict[class] -> value
                    mp = pack.get("mapping", {})
                    mapping_df = pd.DataFrame.from_dict(mp, orient="index")
                    classes = pack.get("classes", list(mapping_df.columns))

                    # —— 列名类型对齐：若 classes 是 [0.0,1.0] 而列名是 ["0.0","1.0"]，做一次重命名
                    rename_map = {}
                    for c in classes:
                        if c not in mapping_df.columns and str(c) in mapping_df.columns:
                            rename_map[str(c)] = c
                    if rename_map:
                        mapping_df = mapping_df.rename(columns=rename_map)

                    # —— 若还有缺失列（某些类训练时没出现），补齐空列，后续 transform 用全局均值兜底
                    for c in classes:
                        if c not in mapping_df.columns:
                            mapping_df[c] = np.nan

                    # —— 固定列顺序为 classes
                    mapping_df = mapping_df[classes]

                    self.state_["tmean"][k] = {
                        "mode": "multiclass",
                        "mapping": mapping_df,
                        "classes": classes,
                        "global": pack.get("global", {}),
                        "alpha": pack.get("alpha", meta.get("alpha", 10)),
                    }
                else:
                    # regression/binary: mapping: dict[category] -> float
                    mp = pack.get("mapping", {})
                    mapping_sr = pd.Series(mp, dtype="float64")
                    self.state_["tmean"][k] = {
                        "mode": mode,
                        "mapping": mapping_sr,
                        "global": pack.get("global", np.nan),
                        "alpha": pack.get("alpha", meta.get("alpha", 10)),
                    }

    # ---------------- 公共 API ----------------
    def fit_transform(self, df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
        out = df.copy()
        if isinstance(self.cfg.get("target_mean"), dict) and self.cfg["target_mean"].get("cols"):
            out = self._fit_transform_target_mean(out, target_col, self.cfg["target_mean"])
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.state_["tmean"]:
            out = self._transform_target_mean(out)
        return out

    def to_dict(self) -> dict:
        # 将 pandas 对象转为基本类型，便于 JSON 保存
        tmean = {}
        for col, pack in self.state_["tmean"].items():
            if col == "_meta":
                tmean[col] = pack
                continue
            mode = pack.get("mode", "regression")
            if mode == "multiclass":
                mapping_df: pd.DataFrame = pack["mapping"]
                tmean[col] = {
                    "mode": "multiclass",
                    "mapping": mapping_df.to_dict(orient="index"),  # {category: {cls: val}}
                    "classes": list(pack.get("classes", mapping_df.columns)),
                    "global": dict(pack.get("global", {})),
                    "alpha": int(pack.get("alpha", 10)),
                }
            else:
                mapping_sr: pd.Series = pack["mapping"]
                tmean[col] = {
                    "mode": mode,
                    "mapping": {k: float(v) for k, v in mapping_sr.to_dict().items()},
                    "global": float(pack.get("global", np.nan)),
                    "alpha": int(pack.get("alpha", 10)),
                }
        bundle = {
            "config": self.cfg,
            "created": list(self.created_),
            "state": {"tmean": tmean},
        }
        return bundle

    # ---------------- TME：实现 ----------------
    def _fit_transform_target_mean(self, df: pd.DataFrame, target_col: str | None, cfg: dict) -> pd.DataFrame:
        if target_col is None or target_col not in df.columns:
            self.log("⚠️ target_mean: 未提供 target_col，跳过。")
            return df

        y = df[target_col]
        nunq = pd.Series(y).nunique(dropna=True)
        cols: List[str] = [c for c in cfg.get("cols", []) if c in df.columns]
        if not cols:
            self.log("⚠️ target_mean: 配置的列不存在，已跳过。")
            return df

        alpha: int = int(cfg.get("alpha", 10))
        min_samples: int = int(cfg.get("min_samples", 1))
        suffix: str = str(cfg.get("suffix", "__tmean"))
        n_splits: int = int(cfg.get("n_splits", 5))
        random_state: int = int(cfg.get("random_state", 42))

        # 判断任务类型
        if pd.api.types.is_numeric_dtype(y) and nunq > 20:
            task = "regression"
        else:
            task = "classification"

        from sklearn.model_selection import KFold, StratifiedKFold
        if task == "classification":
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # 全局均值（回归：float；分类：按类的频率）
        if task == "regression":
            global_mean = float(pd.to_numeric(y, errors="coerce").mean())
        else:
            classes = list(pd.Series(y).dropna().unique())
            # 稳定顺序（可读性）
            try:
                classes = sorted(classes)
            except Exception:
                pass
            # 每个类的全局频率
            global_mean = {}
            for c in classes:
                global_mean[c] = float((y == c).mean())

        self.state_["tmean"]["_meta"] = {
            "alpha": alpha, "n_splits": n_splits, "random_state": random_state,
            "suffix": suffix, "min_samples": min_samples
        }

        for col in cols:
            values = df[col]

            if task == "regression":
                # OOF 单列
                oof = pd.Series(index=df.index, dtype="float64")
                for tr_idx, va_idx in splitter.split(df.drop(columns=[target_col]), y):
                    tr_y = pd.to_numeric(y.iloc[tr_idx], errors="coerce")
                    tr_x = values.iloc[tr_idx]
                    va_x = values.iloc[va_idx]

                    grp = pd.DataFrame({"x": tr_x, "y": tr_y})
                    stats = grp.groupby("x")["y"].agg(["mean", "count"])
                    if min_samples > 1:
                        stats = stats.loc[stats["count"] >= min_samples]
                    smooth = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)

                    mapped = va_x.map(smooth).astype("float64")
                    oof.iloc[va_idx] = mapped.fillna(global_mean).values

                new_col = f"{col}{suffix}"
                df[new_col] = oof
                self.created_.append(new_col)

                # 全量映射
                full_grp = pd.DataFrame({"x": values, "y": pd.to_numeric(y, errors="coerce")})
                full_stats = full_grp.groupby("x")["y"].agg(["mean", "count"])
                if min_samples > 1:
                    full_stats = full_stats.loc[full_stats["count"] >= min_samples]
                full_smooth = (full_stats["mean"] * full_stats["count"] + global_mean * alpha) / (full_stats["count"] + alpha)

                self.state_["tmean"][col] = {
                    "mode": "regression",
                    "mapping": full_smooth.astype("float64"),
                    "global": float(global_mean),
                    "alpha": alpha
                }
                self.log(f"目标均值编码(回归)：col='{col}' -> '{new_col}'，全局均值={global_mean:.6f}，类别数={len(full_smooth)}")

            else:
                # 分类（含二分类 & 多分类）—— 每个类一列（若二分类同样给两列，更通用）
                cls_list = list(global_mean.keys())
                oof_mat = {c: pd.Series(index=df.index, dtype="float64") for c in cls_list}

                for tr_idx, va_idx in splitter.split(df.drop(columns=[target_col]), y):
                    tr_x = values.iloc[tr_idx]
                    va_x = values.iloc[va_idx]
                    tr_y = y.iloc[tr_idx]

                    g = pd.DataFrame({"x": tr_x, "y": tr_y})

                    # 统计每个类别下每个类标签的均值（one-vs-rest 的概率）
                    # 即：对于每个类 c，mean( 1[y==c] ) by x
                    # 然后做平滑
                    count_by_x = g.groupby("x")["y"].count().rename("count")  # 共同的计数
                    # 先构造一个 DataFrame: index=x 类别；列=类标签；值=均值
                    prob_df = []
                    for c in cls_list:
                        bin_mean = (g.assign(ind=(g["y"] == c).astype(int))
                                     .groupby("x")["ind"].mean().rename(c))
                        prob_df.append(bin_mean)
                    prob_df = pd.concat(prob_df, axis=1)  # 有些列可能缺失，后续用 freq 填
                    # 合并计数
                    prob_df = prob_df.join(count_by_x, how="left")

                    # min_samples 过滤
                    if min_samples > 1:
                        prob_df = prob_df.loc[prob_df["count"] >= min_samples]

                    # 平滑： (mean * count + global * alpha) / (count + alpha)，对每个类列分别处理
                    for c in cls_list:
                        gm = float(global_mean[c])
                        col_mean = prob_df[c].fillna(gm)
                        cnt = prob_df["count"].fillna(0.0).astype("float64")
                        smooth = (col_mean * cnt + gm * alpha) / (cnt + alpha)
                        mapped = va_x.map(smooth).astype("float64").fillna(gm)
                        oof_mat[c].iloc[va_idx] = mapped.values

                # 写入新列
                new_cols = []
                for c in cls_list:
                    new_col = f"{col}{suffix}__class_{_slug(c)}"
                    df[new_col] = oof_mat[c]
                    new_cols.append(new_col)
                self.created_.extend(new_cols)

                # 全量映射（供 transform）
                full_g = pd.DataFrame({"x": values, "y": y})
                count_by_x = full_g.groupby("x")["y"].count().rename("count")
                prob_df = []
                for c in cls_list:
                    bin_mean = (full_g.assign(ind=(full_g["y"] == c).astype(int))
                                .groupby("x")["ind"].mean().rename(c))
                    prob_df.append(bin_mean)
                prob_df = pd.concat(prob_df, axis=1)
                prob_df = prob_df.join(count_by_x, how="left")

                if min_samples > 1:
                    prob_df = prob_df.loc[prob_df["count"] >= min_samples]

                smooth_df = pd.DataFrame(index=prob_df.index, columns=cls_list, dtype="float64")
                cnt = prob_df["count"].fillna(0.0).astype("float64")
                for c in cls_list:
                    gm = float(global_mean[c])
                    col_mean = prob_df[c].fillna(gm)
                    smooth_df[c] = (col_mean * cnt + gm * alpha) / (cnt + alpha)
                smooth_df = smooth_df.drop(columns=[c for c in smooth_df.columns if c not in cls_list], errors="ignore")

                self.state_["tmean"][col] = {
                    "mode": "multiclass",
                    "mapping": smooth_df,
                    "classes": cls_list,
                    "global": {k: float(v) for k, v in global_mean.items()},
                    "alpha": alpha
                }
                self.log(f"目标均值编码(多分类)：col='{col}' -> 新增 {len(cls_list)} 列（{', '.join([f'class={_slug(c)}' for c in cls_list])}），类别数={smooth_df.shape[0]}")

        return df

    def _transform_target_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        meta = self.state_["tmean"].get("_meta", {})
        suffix = meta.get("suffix", "__tmean")
        for col, pack in self.state_["tmean"].items():
            if col == "_meta":
                continue
            mode = pack.get("mode", "regression")
            if mode == "multiclass":
                mapping_df: pd.DataFrame = pack["mapping"]
                cls_list: list = pack.get("classes", list(mapping_df.columns))
                gdict: dict = pack.get("global", {})

                if col not in df.columns:
                    # 缺失输入列：直接生成所有新列，用全局均值填充
                    for c in cls_list:
                        new_col = f"{col}{suffix}__class_{_slug(c)}"
                        df[new_col] = float(gdict.get(c, np.nan))
                        if new_col not in self.created_: self.created_.append(new_col)
                    continue

                x = df[col]
                # 为每个类构造列
                for c in cls_list:
                    gm = float(gdict.get(c, np.nan))
                    # 兼容 JSON 列名强转成字符串导致的取列 KeyError
                    try:
                        series_map = mapping_df[c]
                    except KeyError:
                        if str(c) in mapping_df.columns:
                            series_map = mapping_df[str(c)]
                        else:
                            # 兜底：该类对应的列完全缺失（极少见），直接用全局均值填充
                            new_col = f"{col}{suffix}__class_{_slug(c)}"
                            df[new_col] = float(gm)
                            if new_col not in self.created_: self.created_.append(new_col)
                            continue

                    new_col = f"{col}{suffix}__class_{_slug(c)}"
                    df[new_col] = x.map(series_map).astype("float64").fillna(gm)
                    if new_col not in self.created_: self.created_.append(new_col)

            else:
                # 回归/二分类（单列）
                mapping_sr: pd.Series = pack["mapping"]
                g = float(pack.get("global", np.nan))
                new_col = f"{col}{suffix}"
                if col not in df.columns:
                    df[new_col] = g
                else:
                    df[new_col] = df[col].map(mapping_sr).astype("float64").fillna(g)
                if new_col not in self.created_: self.created_.append(new_col)

        return df
