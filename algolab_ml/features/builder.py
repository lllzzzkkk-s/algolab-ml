from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional, Tuple
import json
import pandas as pd
from .ops import (
    add_polynomial, add_interactions, add_bins,
    add_datetime_parts, add_frequency_encode, add_text_basic
)

class FeatureBuilder:
    """
    读取配置 -> 顺序执行各类特征构造；并输出详细日志。
    说明：仅做“派生列构造”，不做缩放/独热（交给你现有 preprocess 步骤）。
    """
    def __init__(self, cfg: Dict[str, Any], log_fn: Optional[Callable[[str], None]] = print):
        self.cfg = cfg or {}
        self.log = log_fn or (lambda *a, **k: None)
        self.created_: List[str] = []

    def _sec(self, title: str):
        self.log(f"\n------- {title} -------")

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series]=None):
        # 目前这些操作不需要拟合参数（target-encoding 另议）
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cur = df.copy()
        self.created_ = []

        # 1) 多项式
        poly = self.cfg.get("polynomial")
        if poly:
            cols = poly.get("cols", [])
            deg = int(poly.get("degree", 2))
            self._sec(f"多项式特征 degree={deg}, 列={cols}")
            cur, added = add_polynomial(cur, cols, degree=deg, include_bias=False)
            self.log(f"新增列数：{len(added)}"); self.created_.extend(added)

        # 2) 交互项
        inter = self.cfg.get("interactions")
        if inter:
            pairs = inter.get("pairs", [])
            self._sec(f"交互项（乘积），对数：{len(pairs)}")
            cur, added = add_interactions(cur, pairs)
            self.log(f"新增列：{added[:8]}{'...' if len(added)>8 else ''}")
            self.created_.extend(added)

        # 3) 分箱
        bin_cfg = self.cfg.get("binning")
        if bin_cfg:
            cols = bin_cfg.get("cols", [])
            method = bin_cfg.get("method", "quantile")
            bins = int(bin_cfg.get("bins", 5))
            self._sec(f"分箱：method={method}, bins={bins}, 列={cols}")
            cur, added = add_bins(cur, cols, method=method, bins=bins)
            self.log(f"新增列：{added}"); self.created_.extend(added)

        # 4) 日期展开
        dparts = self.cfg.get("datetime_parts")
        if dparts:
            self._sec(f"日期展开：{dparts}")
            cur, added = add_datetime_parts(cur, dparts)
            self.log(f"新增列：{added}"); self.created_.extend(added)

        # 5) 频次编码
        freq = self.cfg.get("frequency_encode")
        if freq:
            cols = freq.get("cols", [])
            as_freq = bool(freq.get("as_freq", True))
            self._sec(f"频次编码：cols={cols}, 模式={'freq' if as_freq else 'count'}")
            cur, added = add_frequency_encode(cur, cols, as_freq=as_freq)
            self.log(f"新增列：{added}"); self.created_.extend(added)

        # 6) 文本基础特征
        text = self.cfg.get("text_basic")
        if text:
            cols = text.get("cols", [])
            metrics = text.get("metrics", ["length", "num_alpha", "num_digit"])
            self._sec(f"文本特征：cols={cols}, metrics={metrics}")
            cur, added = add_text_basic(cur, cols, metrics=metrics)
            self.log(f"新增列：{added}"); self.created_.extend(added)

        self._sec("特征工程完成")
        self.log(f"本次新增特征列总数：{len(self.created_)}")
        return cur

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str]=None):
        _ = self.fit(df, y=df[target_col] if (target_col and target_col in df.columns) else None)
        return self.transform(df)

    # 序列化（导出用）
    def to_json(self) -> str:
        return json.dumps(self.cfg, ensure_ascii=False, indent=2)
