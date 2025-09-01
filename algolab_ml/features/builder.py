# algolab_ml/features/builder.py
from __future__ import annotations
import re
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
    统一的特征构建器（包含：
      - target_mean（OOF 防泄漏，支持回归/二分类/多分类）
      - text_tfidf（按列独立的 TF-IDF，训练落盘 vocab+idf，预测复现）
    ）

    cfg 结构示例：
    {
      "target_mean": {
        "cols": ["city","brand"],
        "alpha": 10,
        "min_samples": 2,
        "n_splits": 5,
        "random_state": 42,
        "suffix": "__tmean"
      },
      "text_tfidf": {
        "cols": ["comment"],
        "ngram_range": [1,2],
        "max_features": 200,
        "analyzer": "word",
        "lowercase": true,
        "min_df": 1,
        "max_df": 1.0,
        "stop_words": null,
        "norm": "l2",
        "sublinear_tf": false,
        "prefix": null
      }
    }

    持久化：
    - to_dict() 保存 {"config": cfg, "created": [...], "state": {"tmean": ..., "tfidf": ...}}
    - 纯预测 (--run-dir) 阶段可直接 transform 使用训练期映射/词表
    """

    def __init__(self, cfg_or_bundle: Dict | None = None, log_fn=print):
        cfg_or_bundle = cfg_or_bundle or {}
        if "config" in cfg_or_bundle or "state" in cfg_or_bundle:
            self.cfg: Dict = cfg_or_bundle.get("config", {})
            saved_state = cfg_or_bundle.get("state", {})
        else:
            self.cfg = cfg_or_bundle
            saved_state = {}

        self.log = log_fn
        self.created_: List[str] = []

        # 统一状态容器
        self.state_: Dict = {
            "tmean": {},
            "tfidf": {},  # {<col>: {"params": {...}, "vocabulary": {...}, "idf": [...], "feature_tokens": [...]} }
        }

        # ---------- 恢复 target_mean ----------
        tmean_saved = saved_state.get("tmean", {})
        if tmean_saved:
            meta = tmean_saved.get("_meta", {})
            if meta:
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

                    # 列名类型对齐（兼容 JSON 导致的 str/int 漂移）
                    rename_map = {}
                    for c in classes:
                        if c not in mapping_df.columns and str(c) in mapping_df.columns:
                            rename_map[str(c)] = c
                    if rename_map:
                        mapping_df = mapping_df.rename(columns=rename_map)

                    # 缺失列补齐
                    for c in classes:
                        if c not in mapping_df.columns:
                            mapping_df[c] = np.nan
                    mapping_df = mapping_df[classes]

                    self.state_["tmean"][k] = {
                        "mode": "multiclass",
                        "mapping": mapping_df.astype("float64"),
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
                        "alpha": pack.get("alpha", tmean_saved.get("_meta", {}).get("alpha", 10)),
                    }

        # ---------- 恢复 tfidf ----------
        tfidf_saved = saved_state.get("tfidf", {})
        if tfidf_saved:
            # 直接放入（内部均为 JSON 基本类型）
            for col, pack in tfidf_saved.items():
                if not isinstance(pack, dict):
                    continue
                self.state_["tfidf"][col] = {
                    "params": pack.get("params", {}),
                    "vocabulary": pack.get("vocabulary", {}),
                    "idf": list(pack.get("idf", [])),
                    "feature_tokens": list(pack.get("feature_tokens", [])),
                }

    # ---------------- 公共 API ----------------
    def fit_transform(self, df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
        out = df.copy()

        # 1) 目标均值编码（依赖 target）
        if isinstance(self.cfg.get("target_mean"), dict) and self.cfg["target_mean"].get("cols"):
            out = self._fit_transform_target_mean(out, target_col, self.cfg["target_mean"])

        # 2) 文本 TF-IDF
        cfg_tfidf = self.cfg.get("text_tfidf")
        if isinstance(cfg_tfidf, dict) and cfg_tfidf.get("cols"):
            out = self._fit_transform_text_tfidf(out, cfg_tfidf)

        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # 1) 目标均值编码（使用 state）
        if self.state_.get("tmean"):
            out = self._transform_target_mean(out)

        # 2) 文本 TF-IDF（使用落盘 vocab+idf）
        cfg_tfidf = self.cfg.get("text_tfidf", {})
        # 若 cfg 中没写，但 state 里有，我们也能恢复（取 state 的列清单）
        if self.state_.get("tfidf"):
            out = self._transform_text_tfidf(out, cfg_tfidf if isinstance(cfg_tfidf, dict) else {})

        return out

    def to_dict(self) -> dict:
        # ------ tmean 序列化 ------
        tmean_out = {}
        for col, pack in self.state_["tmean"].items():
            if col == "_meta":
                tmean_out[col] = pack
                continue
            mode = pack.get("mode", "regression")
            if mode == "multiclass":
                mapping_df: pd.DataFrame = pack["mapping"]
                tmean_out[col] = {
                    "mode": "multiclass",
                    "mapping": mapping_df.to_dict(orient="index"),  # {category: {cls: val}}
                    "classes": list(pack.get("classes", mapping_df.columns)),
                    "global": dict(pack.get("global", {})),
                    "alpha": int(pack.get("alpha", 10)),
                }
            else:
                mapping_sr: pd.Series = pack["mapping"]
                tmean_out[col] = {
                    "mode": mode,
                    "mapping": {k: float(v) for k, v in mapping_sr.to_dict().items()},
                    "global": float(pack.get("global", np.nan)),
                    "alpha": int(pack.get("alpha", 10)),
                }

        # ------ tfidf 序列化 ------
        tfidf_out = {}
        for col, pack in self.state_["tfidf"].items():
            tfidf_out[col] = {
                "params": dict(pack.get("params", {})),
                "vocabulary": dict(pack.get("vocabulary", {})),
                "idf": [float(x) for x in pack.get("idf", [])],
                "feature_tokens": list(pack.get("feature_tokens", [])),
            }

        bundle = {
            "config": self.cfg,
            "created": list(self.created_),
            "state": {"tmean": tmean_out, "tfidf": tfidf_out},
        }
        return bundle

    # ---------------- 目标均值编码：实现 ----------------
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
            try:
                classes = sorted(classes)
            except Exception:
                pass
            global_mean = {c: float((y == c).mean()) for c in classes}

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

                    count_by_x = g.groupby("x")["y"].count().rename("count")
                    prob_df = []
                    for c in cls_list:
                        bin_mean = (g.assign(ind=(g["y"] == c).astype(int))
                                    .groupby("x")["ind"].mean().rename(c))
                        prob_df.append(bin_mean)
                    prob_df = pd.concat(prob_df, axis=1)
                    prob_df = prob_df.join(count_by_x, how="left")

                    if min_samples > 1:
                        prob_df = prob_df.loc[prob_df["count"] >= min_samples]

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
                        if new_col not in self.created_:
                            self.created_.append(new_col)
                    continue

                x = df[col]
                for c in cls_list:
                    gm = float(gdict.get(c, np.nan))
                    try:
                        series_map = mapping_df[c]
                    except KeyError:
                        if str(c) in mapping_df.columns:
                            series_map = mapping_df[str(c)]
                        else:
                            new_col = f"{col}{suffix}__class_{_slug(c)}"
                            df[new_col] = float(gm)
                            if new_col not in self.created_:
                                self.created_.append(new_col)
                            continue

                    new_col = f"{col}{suffix}__class_{_slug(c)}"
                    df[new_col] = x.map(series_map).astype("float64").fillna(gm)
                    if new_col not in self.created_:
                        self.created_.append(new_col)

            else:
                mapping_sr: pd.Series = pack["mapping"]
                g = float(pack.get("global", np.nan))
                new_col = f"{col}{suffix}"
                if col not in df.columns:
                    df[new_col] = g
                else:
                    df[new_col] = df[col].map(mapping_sr).astype("float64").fillna(g)
                if new_col not in self.created_:
                    self.created_.append(new_col)

        return df

    # ---------------- 文本 TF-IDF：实现 ----------------
    def _fit_transform_text_tfidf(self, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """
        拟合 + 产出 TF-IDF 特征（逐列独立建模，便于落盘复现）
        """
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

        cols = [c for c in cfg.get("cols", []) if c in df.columns]
        if not cols:
            return df

        # 统一参数
        ng = tuple(cfg.get("ngram_range", [1, 1]))
        max_feat = int(cfg.get("max_features", 200))
        analyzer = cfg.get("analyzer", "word")
        lowercase = bool(cfg.get("lowercase", True))
        min_df = cfg.get("min_df", 1)
        max_df = cfg.get("max_df", 1.0)
        stop_words = cfg.get("stop_words", None)
        norm = cfg.get("norm", "l2")
        sublinear_tf = bool(cfg.get("sublinear_tf", False))
        prefix_opt = cfg.get("prefix", None)

        out = df.copy()

        for col in cols:
            text = out[col].fillna("").astype(str).values

            # 先用 CountVectorizer 拟合词袋
            cv = CountVectorizer(
                ngram_range=ng,
                max_features=max_feat,
                analyzer=analyzer,
                lowercase=lowercase,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words
            )
            X_counts = cv.fit_transform(text)

            # 再用 TfidfTransformer 拟合 IDF
            tfidf = TfidfTransformer(norm=norm, use_idf=True, smooth_idf=True, sublinear_tf=sublinear_tf)
            X_tfidf = tfidf.fit_transform(X_counts)

            # 列名
            feat_names = [f"{prefix_opt or (col + '__tfidf')}__{t}".replace(" ", "_") for t in cv.get_feature_names_out()]

            # 低维时直接转 dense；太大则用 pandas 稀疏
            if X_tfidf.shape[1] <= 2000:
                dense = X_tfidf.toarray()
                tfidf_df = pd.DataFrame(dense, columns=feat_names, index=out.index)
            else:
                tfidf_df = pd.DataFrame.sparse.from_spmatrix(X_tfidf, index=out.index, columns=feat_names)

            out = pd.concat([out, tfidf_df], axis=1)
            # 记录新列
            self.created_.extend([c for c in tfidf_df.columns if c not in self.created_])

            # —— 将可复现的状态落入 state（纯 JSON 友好）
            self.state_["tfidf"][col] = {
                "params": {
                    "ngram_range": list(ng),
                    "analyzer": analyzer,
                    "lowercase": lowercase,
                    "min_df": min_df,
                    "max_df": max_df,
                    "stop_words": stop_words,
                    "norm": norm,
                    "sublinear_tf": sublinear_tf,
                    "max_features": max_feat,
                    "prefix": prefix_opt,
                },
                "vocabulary": cv.vocabulary_,              # {token: index}
                "idf": tfidf.idf_.tolist(),                # [idf_i ...] 与 vocabulary 索引对齐
                "feature_tokens": cv.get_feature_names_out().tolist(),
            }

            self.log(f"TF-IDF: col='{col}'，tokens={len(feat_names)}")

        return out

    def _transform_text_tfidf(self, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """
        仅 transform：用落盘的 vocab + idf 复现 TF-IDF。
        若 cfg 未提供 cols，则使用 state["tfidf"].keys()。
        """
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

        # 优先用 cfg.cols；否则用 state 中的列
        cfg_cols = [c for c in cfg.get("cols", []) if c in df.columns] if isinstance(cfg, dict) else []
        state_cols = list(self.state_.get("tfidf", {}).keys())
        cols = cfg_cols or state_cols
        if not cols:
            return df

        out = df.copy()

        for col in cols:
            state = self.state_.get("tfidf", {}).get(col)
            if not state:
                # 没有拟合过，跳过
                continue

            p = state["params"]
            vocab = state["vocabulary"]
            idf = np.array(state["idf"], dtype="float64")
            tokens = state.get("feature_tokens", [])

            # 用固定词表的 CountVectorizer
            cv = CountVectorizer(
                ngram_range=tuple(p.get("ngram_range", [1, 1])),
                analyzer=p.get("analyzer", "word"),
                lowercase=bool(p.get("lowercase", True)),
                min_df=1, max_df=1.0,
                stop_words=p.get("stop_words", None),
                vocabulary=vocab
            )
            X_counts = cv.transform(out[col].fillna("").astype(str).values)

            # 复原 IDF
            tfidf = TfidfTransformer(
                norm=p.get("norm", "l2"),
                use_idf=True, smooth_idf=True,
                sublinear_tf=bool(p.get("sublinear_tf", False))
            )
            tfidf.idf_ = idf

            X_tfidf = tfidf.transform(X_counts)
            feat_names = [f"{p.get('prefix') or (col + '__tfidf')}__{t}".replace(" ", "_") for t in tokens]

            if X_tfidf.shape[1] <= 2000:
                dense = X_tfidf.toarray()
                tfidf_df = pd.DataFrame(dense, columns=feat_names, index=out.index)
            else:
                tfidf_df = pd.DataFrame.sparse.from_spmatrix(X_tfidf, index=out.index, columns=feat_names)

            out = pd.concat([out, tfidf_df], axis=1)
            self.created_.extend([c for c in tfidf_df.columns if c not in self.created_])

        return out
