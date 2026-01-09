"""
GPU 批量因子评估模块

提供 GPU 加速的因子评估功能：
- 批量市值中性化
- 批量行业中性化
- 批量 IC/单调性计算
- CPU fallback 当 GPU 不可用时
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import qlib
from qlib.data import D
from qlib.config import C

from factor_env import (
    _load_csi500_membership,
    _align_membership_index,
    DEFAULT_CSI500_MEMBERSHIP,
    expand_vwap_expressions,
)


class GPUFactorEvaluator:
    """
    GPU 加速的批量因子评估器
    
    特性：
    - 批量处理多个因子（默认 batch_size=16）
    - GPU 加速的市值/行业中性化
    - 自动 CPU fallback
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        provider_uri: str = "./qlib_bin_data",
        csi500_membership_path: Optional[str] = None,
        batch_size: int = 16,
        device: Optional[torch.device] = None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.provider_uri = provider_uri
        self.csi500_membership_path = csi500_membership_path or str(DEFAULT_CSI500_MEMBERSHIP)
        self.batch_size = batch_size
        
        # 设备选择：优先 GPU，fallback 到 CPU
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = torch.device("xpu")
        else:
            self.device = torch.device("cpu")
        
        self.use_gpu = self.device.type != "cpu"
        print(f"[GPUFactorEvaluator] Using device: {self.device} (GPU accelerated: {self.use_gpu})")
        
        # 初始化 Qlib
        if not C.get("initialized", False):
            qlib.init(provider_uri=provider_uri, region=qlib.constant.REG_CN)
        
        # 预加载数据
        self._preload_data()
    
    def _preload_data(self):
        """预加载目标收益、市值、行业等数据到 GPU"""
        print("[GPUFactorEvaluator] Preloading data...")
        
        # 加载目标收益
        target_df = D.features(
            D.instruments("all"),
            ["Ref($close, -5) / Ref($close, -1) - 1"],
            start_time=self.start_date,
            end_time=self.end_date,
            freq="day",
        )
        target_df.columns = ["target"]
        
        # 加载市值和行业
        base_df = D.features(
            D.instruments("all"),
            ["$close", "$industry", "$circ_mv"],
            start_time=self.start_date,
            end_time=self.end_date,
            freq="day",
        )
        
        # 合并数据
        merged_df = pd.merge(base_df, target_df, left_index=True, right_index=True, how="inner")
        
        # 过滤掉指数
        if "SH000300" in merged_df.index.get_level_values("instrument"):
            merged_df = merged_df.drop("SH000300", level="instrument")
        
        # 获取唯一的日期和股票列表
        self.dates = merged_df.index.get_level_values("datetime").unique().sort_values()
        self.instruments = merged_df.index.get_level_values("instrument").unique().sort_values()
        
        self.num_dates = len(self.dates)
        self.num_stocks = len(self.instruments)
        
        # 创建索引映射
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.inst_to_idx = {inst: i for i, inst in enumerate(self.instruments)}
        
        # 先用 numpy 创建数据数组，然后一次性转换为 GPU 张量
        target_np = np.full((self.num_dates, self.num_stocks), np.nan, dtype=np.float32)
        log_mkt_cap_np = np.full((self.num_dates, self.num_stocks), np.nan, dtype=np.float32)
        industry_np = np.full((self.num_dates, self.num_stocks), -1, dtype=np.int64)
        close_np = np.full((self.num_dates, self.num_stocks), np.nan, dtype=np.float32)
        
        # 填充数据
        for (inst, dt), row in merged_df.iterrows():
            if inst not in self.inst_to_idx or dt not in self.date_to_idx:
                continue
            d_idx = self.date_to_idx[dt]
            s_idx = self.inst_to_idx[inst]
            
            target_val = row["target"]
            if not pd.isna(target_val):
                target_np[d_idx, s_idx] = float(target_val)
            
            close_val = row["$close"]
            if not pd.isna(close_val):
                close_np[d_idx, s_idx] = float(close_val)
            
            circ_mv = row["$circ_mv"]
            if not pd.isna(circ_mv) and circ_mv > 0:
                log_mkt_cap_np[d_idx, s_idx] = float(np.log(circ_mv + 1))
            
            industry = row["$industry"]
            if not pd.isna(industry):
                industry_np[d_idx, s_idx] = int(industry)
        
        # 转换为 PyTorch 张量并移动到 GPU
        self.target_tensor = torch.from_numpy(target_np).to(self.device)
        self.log_mkt_cap_tensor = torch.from_numpy(log_mkt_cap_np).to(self.device)
        self.industry_tensor = torch.from_numpy(industry_np).to(self.device)
        self.close_tensor = torch.from_numpy(close_np).to(self.device)
        
        # 创建有效数据 mask
        self.valid_mask = ~torch.isnan(self.target_tensor) & ~torch.isnan(self.log_mkt_cap_tensor)
        
        # 行业数量
        self.num_industries = self.industry_tensor.max().item() + 1
        
        # CSI500 成分股 mask
        self._build_csi500_mask(merged_df.index)
        
        print(f"[GPUFactorEvaluator] Loaded {self.num_dates} dates x {self.num_stocks} stocks")
        print(f"[GPUFactorEvaluator] Valid data points: {self.valid_mask.sum().item()}")
    
    def _build_csi500_mask(self, original_index: pd.MultiIndex):
        """构建 CSI500 成分股 mask"""
        csi500_index = _load_csi500_membership(self.csi500_membership_path)
        self.csi500_mask = torch.zeros((self.num_dates, self.num_stocks), dtype=torch.bool, device=self.device)
        
        if csi500_index is None:
            # 如果没有 CSI500 数据，默认全选
            self.csi500_mask.fill_(True)
            return
        
        in_membership = _align_membership_index(original_index, csi500_index)
        
        for (inst, dt), is_member in zip(original_index, in_membership):
            if inst not in self.inst_to_idx or dt not in self.date_to_idx:
                continue
            if is_member:
                d_idx = self.date_to_idx[dt]
                s_idx = self.inst_to_idx[inst]
                self.csi500_mask[d_idx, s_idx] = True
    
    def _load_factor_batch(self, factor_expressions: List[str]) -> torch.Tensor:
        """
        加载一批因子数据并转换为张量
        
        Returns:
            factors: [batch_size, num_dates, num_stocks]
        """
        batch_size = len(factor_expressions)
        
        # 先用 numpy 创建，避免 CUDA 张量赋值问题
        factors_np = np.full(
            (batch_size, self.num_dates, self.num_stocks), 
            np.nan, 
            dtype=np.float32
        )
        
        # 展开 VWAP 表达式
        expanded_exprs, rename_map = expand_vwap_expressions(factor_expressions)
        
        try:
            factor_df = D.features(
                D.instruments("all"),
                expanded_exprs,
                start_time=self.start_date,
                end_time=self.end_date,
                freq="day",
            )
        except Exception as e:
            print(f"[GPUFactorEvaluator] Error loading factors: {e}")
            return torch.from_numpy(factors_np).to(self.device)
        
        if factor_df.empty:
            return torch.from_numpy(factors_np).to(self.device)
        
        # 重命名列
        if rename_map:
            factor_df = factor_df.rename(columns=rename_map)
        
        # 过滤指数
        if "SH000300" in factor_df.index.get_level_values("instrument"):
            factor_df = factor_df.drop("SH000300", level="instrument")
        
        # 填充 numpy 数组
        for (inst, dt), row in factor_df.iterrows():
            if inst not in self.inst_to_idx or dt not in self.date_to_idx:
                continue
            d_idx = self.date_to_idx[dt]
            s_idx = self.inst_to_idx[inst]
            
            for i, expr in enumerate(factor_expressions):
                val = row[expr] if expr in row.index else np.nan
                if not pd.isna(val):
                    factors_np[i, d_idx, s_idx] = float(val)
        
        # 转换为 GPU 张量
        return torch.from_numpy(factors_np).to(self.device)
    
    def _clip_outliers_gpu(self, factors: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        GPU 批量去极值（3 sigma）
        
        Args:
            factors: [B, D, S]
            valid_mask: [D, S]
        
        Returns:
            clipped: [B, D, S]
        """
        B, D, S = factors.shape
        mask = valid_mask.unsqueeze(0).expand(B, -1, -1)  # [B, D, S]
        
        # 替换无效值为 0 进行计算
        factors_masked = factors.clone()
        factors_masked[~mask | torch.isnan(factors)] = 0
        
        # 每日计算均值和标准差
        count = mask.float().sum(dim=2, keepdim=True).clamp(min=1)  # [B, D, 1]
        mean = factors_masked.sum(dim=2, keepdim=True) / count  # [B, D, 1]
        
        sq_diff = ((factors_masked - mean) ** 2) * mask.float()
        std = torch.sqrt(sq_diff.sum(dim=2, keepdim=True) / count.clamp(min=1))  # [B, D, 1]
        
        # Clip
        lower = mean - 3 * std
        upper = mean + 3 * std
        clipped = torch.clamp(factors, lower, upper)
        
        # 保持原始 NaN
        clipped[torch.isnan(factors)] = 0  # 填充为 0
        
        return clipped
    
    def _regress_out_size_gpu(self, factors: torch.Tensor) -> torch.Tensor:
        """
        GPU 批量市值中性化
        
        对每个日期，对因子 Y 和对数市值 X 做回归：
        Y_residual = Y - (slope * X + intercept)
        
        Args:
            factors: [B, D, S] 已去极值的因子
        
        Returns:
            residuals: [B, D, S] 残差
        """
        B, D, S = factors.shape
        
        X = self.log_mkt_cap_tensor.unsqueeze(0)  # [1, D, S]
        Y = factors  # [B, D, S]
        
        # 有效 mask：因子和市值都有效
        factor_valid = ~torch.isnan(factors)
        base_valid = self.valid_mask.unsqueeze(0).expand(B, -1, -1)  # [B, D, S]
        mask = factor_valid & base_valid
        
        # 计数
        count = mask.float().sum(dim=2, keepdim=True).clamp(min=1)  # [B, D, 1]
        
        # 将无效位置设为 0
        X_safe = X.clone()
        X_safe[~self.valid_mask.unsqueeze(0)] = 0
        Y_safe = factors.clone()
        Y_safe[~mask] = 0
        
        # 均值
        X_mean = (X_safe * mask.float()).sum(dim=2, keepdim=True) / count  # [1, D, 1]
        Y_mean = (Y_safe * mask.float()).sum(dim=2, keepdim=True) / count  # [B, D, 1]
        
        # 中心化
        X_centered = (X_safe - X_mean) * mask.float()  # [1, D, S]
        Y_centered = (Y_safe - Y_mean) * mask.float()  # [B, D, S]
        
        # 斜率
        numerator = (X_centered * Y_centered).sum(dim=2, keepdim=True)  # [B, D, 1]
        denominator = (X_centered ** 2).sum(dim=2, keepdim=True).clamp(min=1e-8)  # [1, D, 1]
        slope = numerator / denominator  # [B, D, 1]
        
        # 截距
        intercept = Y_mean - slope * X_mean  # [B, D, 1]
        
        # 残差
        residual = Y_safe - (slope * X_safe + intercept)  # [B, D, S]
        residual = residual * mask.float()  # 无效位置设为 0
        
        return residual
    
    def _industry_neutralize_gpu(self, factors: torch.Tensor) -> torch.Tensor:
        """
        GPU 批量行业中性化 + 标准化
        
        使用 scatter 操作实现 groupby(['datetime', 'industry']).transform(z-score)
        
        Args:
            factors: [B, D, S] 市值中性化后的因子
        
        Returns:
            neutralized: [B, D, S] 行业中性化后的因子
        """
        B, D, S = factors.shape
        device = factors.device
        
        # 有效 mask
        factor_valid = factors != 0  # 之前已将无效值设为 0
        industry_valid = self.industry_tensor >= 0
        mask = factor_valid & industry_valid.unsqueeze(0).expand(B, -1, -1)
        
        # 构建 group_id: [D, S]
        # group_id = date_idx * num_industries + industry_id
        date_idx = torch.arange(D, device=device).unsqueeze(1)  # [D, 1]
        industry_ids = self.industry_tensor.clamp(min=0)  # [D, S], 确保非负
        group_id = date_idx * self.num_industries + industry_ids  # [D, S]
        num_groups = D * self.num_industries
        
        # 展平
        factors_flat = factors.view(B, -1)  # [B, D*S]
        group_id_flat = group_id.view(-1)  # [D*S]
        mask_flat = mask.view(B, -1)  # [B, D*S]
        
        # 为无效位置指定一个 "垃圾桶" group
        valid_group_id = group_id_flat.unsqueeze(0).expand(B, -1).clone()  # [B, D*S]
        valid_group_id[~mask_flat] = num_groups  # 无效指向 num_groups
        
        # scatter_add 计算每组的 sum 和 sum of squares
        group_sum = torch.zeros(B, num_groups + 1, device=device)
        group_sq_sum = torch.zeros(B, num_groups + 1, device=device)
        group_count = torch.zeros(B, num_groups + 1, device=device)
        
        factors_masked = factors_flat * mask_flat.float()
        
        group_sum.scatter_add_(1, valid_group_id, factors_masked)
        group_sq_sum.scatter_add_(1, valid_group_id, factors_masked ** 2)
        group_count.scatter_add_(1, valid_group_id, mask_flat.float())
        
        # 计算均值和标准差（只取前 num_groups 个，忽略垃圾桶）
        group_count = group_count[:, :num_groups].clamp(min=1)  # [B, num_groups]
        group_sum = group_sum[:, :num_groups]  # [B, num_groups]
        group_sq_sum = group_sq_sum[:, :num_groups]  # [B, num_groups]
        
        group_mean = group_sum / group_count  # [B, num_groups]
        group_var = group_sq_sum / group_count - group_mean ** 2
        group_std = torch.sqrt(group_var.clamp(min=1e-8))  # [B, num_groups]
        
        # 将 group_id 的无效位置 clamp 到有效范围内（用于 gather）
        gather_idx = group_id_flat.unsqueeze(0).expand(B, -1).clamp(0, num_groups - 1)  # [B, D*S]
        
        mean_per_sample = torch.gather(group_mean, 1, gather_idx)  # [B, D*S]
        std_per_sample = torch.gather(group_std, 1, gather_idx)  # [B, D*S]
        
        # Z-score 标准化
        normalized = (factors_flat - mean_per_sample) / std_per_sample
        normalized = normalized * mask_flat.float()  # 无效位置设为 0
        
        return normalized.view(B, D, S)
    
    def _compute_price_corr_gpu(self, factors: torch.Tensor) -> torch.Tensor:
        """
        GPU 计算因子与价格的 Spearman 相关系数
        
        Args:
            factors: [B, D, S] 原始因子（未中性化）
        
        Returns:
            price_corr: [B] 平均价格相关系数
        """
        B, D, S = factors.shape
        
        # CSI500 mask
        analysis_mask = self.valid_mask & self.csi500_mask  # [D, S]
        analysis_mask = analysis_mask.unsqueeze(0).expand(B, -1, -1) & ~torch.isnan(factors)  # [B, D, S]
        
        daily_corrs = []
        
        for d in range(D):
            day_mask = analysis_mask[:, d, :]  # [B, S]
            day_factors = factors[:, d, :]  # [B, S]
            day_close = self.close_tensor[d, :]  # [S]
            
            # 每个 batch 的相关系数
            batch_corrs = []
            for b in range(B):
                m = day_mask[b]
                if m.sum() < 10:
                    batch_corrs.append(float('nan'))
                    continue
                
                f = day_factors[b][m]
                c = day_close[m]
                
                # Spearman = Pearson on ranks
                f_rank = f.argsort().argsort().float()
                c_rank = c.argsort().argsort().float()
                
                f_centered = f_rank - f_rank.mean()
                c_centered = c_rank - c_rank.mean()
                
                corr = (f_centered * c_centered).sum() / (f_centered.norm() * c_centered.norm() + 1e-8)
                batch_corrs.append(corr.item())
            
            daily_corrs.append(batch_corrs)
        
        # [D, B] -> [B]
        daily_corrs = torch.tensor(daily_corrs, device=self.device)  # [D, B]
        price_corr = daily_corrs.nanmean(dim=0)  # [B]
        
        # NaN 处理
        price_corr = torch.where(torch.isnan(price_corr), torch.ones_like(price_corr), price_corr)
        
        return price_corr
    
    def _compute_monotonicity_gpu(self, factors: torch.Tensor) -> torch.Tensor:
        """
        GPU 计算因子单调性（分五组，看收益是否单调递增）
        
        Args:
            factors: [B, D, S] 中性化后的因子
        
        Returns:
            monotonicity: [B] 平均单调性
        """
        B, D, S = factors.shape
        
        # CSI500 mask
        analysis_mask = self.valid_mask & self.csi500_mask  # [D, S]
        
        daily_mono = []
        
        for d in range(D):
            day_mask = analysis_mask[d, :]  # [S]
            day_target = self.target_tensor[d, :]  # [S]
            
            batch_mono = []
            for b in range(B):
                day_factor = factors[b, d, :]  # [S]
                
                # 有效数据
                valid = day_mask & (day_factor != 0) & ~torch.isnan(day_target)
                if valid.sum() < 25:  # 至少每组 5 个
                    batch_mono.append(float('nan'))
                    continue
                
                f = day_factor[valid]
                t = day_target[valid]
                
                # 分五组
                n = len(f)
                sorted_idx = f.argsort()
                group_size = n // 5
                
                group_returns = []
                for g in range(5):
                    start_idx = g * group_size
                    end_idx = (g + 1) * group_size if g < 4 else n
                    group_indices = sorted_idx[start_idx:end_idx]
                    group_ret = t[group_indices].mean()
                    group_returns.append(group_ret.item())
                
                # Spearman 与 [0,1,2,3,4]
                ranks = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=self.device)
                rets = torch.tensor(group_returns, device=self.device)
                
                r_centered = ranks - ranks.mean()
                ret_centered = rets - rets.mean()
                
                mono = (r_centered * ret_centered).sum() / (r_centered.norm() * ret_centered.norm() + 1e-8)
                batch_mono.append(mono.item())
            
            daily_mono.append(batch_mono)
        
        # [D, B] -> [B]
        daily_mono = torch.tensor(daily_mono, device=self.device)  # [D, B]
        monotonicity = daily_mono.nanmean(dim=0)  # [B]
        
        # NaN 处理，默认 0
        monotonicity = torch.where(torch.isnan(monotonicity), torch.zeros_like(monotonicity), monotonicity)
        
        return monotonicity
    
    def evaluate_factors_batch(
        self, 
        factor_expressions: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """
        批量评估多个因子
        
        Args:
            factor_expressions: 因子表达式列表
        
        Returns:
            results: {expr: (price_corr, monotonicity)}
        """
        if not factor_expressions:
            return {}
        
        batch_size = len(factor_expressions)
        print(f"[GPUFactorEvaluator] Evaluating {batch_size} factors...")
        
        # 1. 加载因子数据
        factors = self._load_factor_batch(factor_expressions)  # [B, D, S]
        
        # 检查是否有有效数据
        valid_factors = ~torch.isnan(factors)
        if valid_factors.sum() == 0:
            print("[GPUFactorEvaluator] No valid factor data")
            return {expr: (None, None) for expr in factor_expressions}
        
        # 2. 计算价格相关系数（用原始因子）
        price_corr = self._compute_price_corr_gpu(factors)  # [B]
        
        # 3. 去极值
        factors_clipped = self._clip_outliers_gpu(factors, self.valid_mask)  # [B, D, S]
        
        # 4. 市值中性化
        factors_size_neu = self._regress_out_size_gpu(factors_clipped)  # [B, D, S]
        
        # 5. 行业中性化
        factors_neu = self._industry_neutralize_gpu(factors_size_neu)  # [B, D, S]
        
        # 6. 计算单调性
        monotonicity = self._compute_monotonicity_gpu(factors_neu)  # [B]
        
        # 组装结果
        results = {}
        for i, expr in enumerate(factor_expressions):
            pc = price_corr[i].item() if not torch.isnan(price_corr[i]) else None
            mono = monotonicity[i].item() if not torch.isnan(monotonicity[i]) else None
            results[expr] = (pc, mono)
        
        return results
    
    def evaluate_single_factor(
        self, 
        factor_expression: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        评估单个因子（便捷方法）
        
        Returns:
            (price_corr, monotonicity)
        """
        results = self.evaluate_factors_batch([factor_expression])
        return results.get(factor_expression, (None, None))


# ========== CPU Fallback 函数 ==========

def evaluate_factor_quality_cpu(
    factor_expression: str,
    start_date: str,
    end_date: str,
    provider_uri: str = "./qlib_bin_data",
    csi500_membership_path=DEFAULT_CSI500_MEMBERSHIP,
) -> Tuple[Optional[float], Optional[float]]:
    """
    CPU 版本的因子质量评估（fallback）
    直接调用 linear_model.py 中的实现
    """
    from linear_model import evaluate_factor_quality
    return evaluate_factor_quality(
        factor_expression, 
        start_date, 
        end_date, 
        provider_uri, 
        csi500_membership_path
    )


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 测试 GPU 评估器
    evaluator = GPUFactorEvaluator(
        start_date="2023-01-01",
        end_date="2024-01-01",
        batch_size=16,
    )
    
    # 测试单个因子
    expr = "Div($close, Ref($close, 5))"
    price_corr, mono = evaluator.evaluate_single_factor(expr)
    print(f"Single factor: price_corr={price_corr:.4f}, monotonicity={mono:.4f}")
    
    # 测试批量因子
    exprs = [
        "Div($close, Ref($close, 5))",
        "Sub($high, $low)",
        "Div($volume, Ref($volume, 5))",
    ]
    results = evaluator.evaluate_factors_batch(exprs)
    for expr, (pc, mono) in results.items():
        print(f"{expr}: price_corr={pc:.4f}, monotonicity={mono:.4f}")
