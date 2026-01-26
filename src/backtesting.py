"""
경로: src/backtesting.py
백테스팅 모듈 (타이밍 동기화 수정 - 인덱스 에러 해결)

핵심 수정사항:
1. dates 반환 시 정확한 수익률 발생 날짜와 매칭
2. portfolio_values와 dates의 길이 일치 보장
3. pandas Series 인덱싱 문제 해결 (.iloc 사용)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import matplotlib.pyplot as plt


class TrustRegionRebalancer:
    """Trust-Region 기반 점진적 리밸런싱"""

    def __init__(
        self,
        min_weight: float = 0.05,
        max_weight: float = 0.35,
        trust_region: float = 0.15,
        action_scaling: float = 1.5,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.trust_region = trust_region
        self.action_scaling = action_scaling

    def action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Action → Weights 변환"""
        action = np.asarray(action).reshape(-1)
        scaled_action = self.action_scaling * action
        exp_scaled = np.exp(scaled_action - np.max(scaled_action))
        weights = exp_scaled / exp_scaled.sum()
        return weights

    def rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> np.ndarray:
        """Trust-Region 리밸런싱"""
        current_weights = np.asarray(current_weights).reshape(-1)
        target_weights = np.asarray(target_weights).reshape(-1)

        delta = target_weights - current_weights
        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.trust_region:
            delta = delta * (self.trust_region / delta_norm)

        new_weights = current_weights + delta
        new_weights = np.clip(new_weights, self.min_weight, self.max_weight)
        new_weights = new_weights / new_weights.sum()
        return new_weights


class Backtester:
    """
    단일 DRL 에이전트용 백테스터 (타이밍 동기화 수정)
    
    핵심 로직:
    - t 시점: state[t]를 보고 weights[t] 결정
    - t→t+1 기간: weights[t]로 returns[t+1] 실현
    - portfolio_value[t+1]: t+1 시점의 포트폴리오 가치
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        dates: pd.Series,
        rebalancer: TrustRegionRebalancer,
        transaction_cost: float = 0.001,
        rebalance_freq: int = 4,
    ):
        self.returns = returns.values
        self.dates = dates if isinstance(dates, pd.Series) else pd.Series(dates)
        self.rebalancer = rebalancer
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq
        self.n_assets = self.returns.shape[1]

    def run(
        self,
        agent,
        states: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> Dict:
        """
        백테스트 실행
        
        타이밍 구조:
        - Loop: t = start_idx ... end_idx-1
        - t 시점에 state[t]로 weights[t] 결정
        - weights[t]는 returns[t+1]에 적용됨
        - portfolio_values[0] = 1.0 (t=start_idx 시작 시점)
        - portfolio_values[k] = t=start_idx+k 시점의 가치
        
        반환되는 dates:
        - dates[0] = start_idx 날짜 (초기값)
        - dates[k] = start_idx+k 날짜 (k번째 포트폴리오 가치의 날짜)
        - 길이 = portfolio_values와 동일
        """
        portfolio_values = [1.0]  # 초기 포트폴리오 가치
        weights_history = []
        current_weights = np.ones(self.n_assets) / self.n_assets

        print("\nRunning backtest...")
        print(f"  Backtest period: index {start_idx} to {end_idx-1}")
        print(f"  Date range: {self.dates.iloc[start_idx]} to {self.dates.iloc[end_idx-1]}")

        for t in range(start_idx, end_idx - 1):
            # 1. t 시점의 상태로 행동 결정
            state = states[t]
            raw_action = agent.predict(state, deterministic=True)
            target_weights = self.rebalancer.action_to_weights(raw_action)

            # 2. 리밸런싱 여부 결정
            step_from_start = t - start_idx
            if step_from_start % self.rebalance_freq == 0:
                new_weights = self.rebalancer.rebalance(
                    current_weights, target_weights
                )
                weight_change = np.abs(new_weights - current_weights).sum()
                trading_cost = weight_change * self.transaction_cost
            else:
                new_weights = current_weights
                trading_cost = 0.0

            # 3. t→t+1 기간 수익률 실현
            period_return = np.dot(new_weights, self.returns[t + 1])
            net_return = period_return - trading_cost

            # 4. t+1 시점 포트폴리오 가치 업데이트
            portfolio_values.append(portfolio_values[-1] * (1 + net_return))
            weights_history.append(new_weights.copy())
            current_weights = new_weights

        # 🔧 핵심 수정: dates는 portfolio_values와 정확히 매칭
        # portfolio_values[0] = start_idx 날짜의 가치
        # portfolio_values[k] = start_idx+k 날짜의 가치
        # 🔧 .iloc 사용하여 위치 기반 인덱싱
        backtest_dates = self.dates.iloc[start_idx:start_idx + len(portfolio_values)].reset_index(drop=True)

        print(f"\n  Backtest completed:")
        print(f"    Portfolio values length: {len(portfolio_values)}")
        print(f"    Weights history length: {len(weights_history)}")
        print(f"    Dates length: {len(backtest_dates)}")
        print(f"    First date: {backtest_dates.iloc[0]}")
        print(f"    Last date: {backtest_dates.iloc[-1]}")

        results = {
            "portfolio_values": np.array(portfolio_values),
            "weights": np.array(weights_history),
            "dates": backtest_dates,
        }
        results["metrics"] = self._calculate_metrics(results["portfolio_values"])
        
        return results

    def _calculate_metrics(self, portfolio_values: np.ndarray) -> Dict:
        """성과 지표 계산"""
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        n_weeks = len(returns)
        n_years = n_weeks / 52 if n_weeks > 0 else 0.0001

        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        annualized_return = ((portfolio_values[-1] / portfolio_values[0]) ** (1 / n_years) - 1) * 100
        annualized_vol = np.std(returns) * np.sqrt(52) * 100
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdowns) * 100

        win_rate = np.sum(returns > 0) / len(returns) * 100 if len(returns) > 0 else 0.0

        return {
            "Total Return (%)": total_return,
            "Annualized Return (%)": annualized_return,
            "Annualized Volatility (%)": annualized_vol,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown (%)": max_drawdown,
            "Win Rate (%)": win_rate,
        }

    def print_metrics(self, metrics: Dict) -> None:
        """성과 지표 출력"""
        print("\nBacktest Performance Metrics")
        print("-" * 40)
        for k, v in metrics.items():
            if "Ratio" in k:
                print(f"{k:30s}: {v:.3f}")
            else:
                print(f"{k:30s}: {v:.2f}")


class EnsembleBacktester:
    """앙상블 백테스터 (타이밍 동기화 수정)"""

    def __init__(
        self,
        returns: pd.DataFrame,
        dates: pd.Series,
        rebalancer: TrustRegionRebalancer,
        transaction_cost: float = 0.001,
        rebalance_freq: int = 4,
    ):
        self.returns = returns.values
        self.dates = dates if isinstance(dates, pd.Series) else pd.Series(dates)
        self.rebalancer = rebalancer
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq
        self.n_assets = self.returns.shape[1]

    def run_sliding_window(
        self,
        window_models: Dict,
        states: np.ndarray,
        valid_indices: np.ndarray,
        slide_step: int = 26,
    ) -> Dict:
        """슬라이딩 윈도우 앙상블 백테스트"""
        n_valid = len(valid_indices)
        portfolio_values = [1.0]
        weights_history = []
        current_weights = np.ones(self.n_assets) / self.n_assets

        print(f"\nRunning ensemble backtest...")

        for t in range(n_valid - 1):
            actual_idx = valid_indices[t]

            # 윈도우 모델 선택
            window_idx = t // slide_step
            if window_idx >= len(window_models):
                window_idx = len(window_models) - 1

            if window_idx in window_models:
                models = window_models[window_idx]["models"]
            else:
                window_idx = max(window_models.keys())
                models = window_models[window_idx]["models"]

            # 앙상블 예측
            state = states[actual_idx]
            ensemble_weights = []

            for algo, agent in models.items():
                raw_action = agent.predict(state, deterministic=True)
                weights = self.rebalancer.action_to_weights(raw_action)
                ensemble_weights.append(weights)

            target_weights = np.mean(ensemble_weights, axis=0)

            # 리밸런싱
            if t % self.rebalance_freq == 0:
                new_weights = self.rebalancer.rebalance(current_weights, target_weights)
                weight_change = np.abs(new_weights - current_weights).sum()
                trading_cost = weight_change * self.transaction_cost
            else:
                new_weights = current_weights
                trading_cost = 0.0

            # 수익률 실현
            period_return = np.dot(new_weights, self.returns[actual_idx + 1])
            net_return = period_return - trading_cost

            portfolio_values.append(portfolio_values[-1] * (1 + net_return))
            weights_history.append(new_weights.copy())
            current_weights = new_weights

        # 🔧 수정: dates 정확히 매칭 (.iloc 사용)
        start_idx = valid_indices[0]
        backtest_dates = self.dates.iloc[start_idx:start_idx + len(portfolio_values)].reset_index(drop=True)

        results = {
            "portfolio_values": np.array(portfolio_values),
            "weights": np.array(weights_history),
            "dates": backtest_dates,
        }

        results["metrics"] = self._calculate_metrics(results["portfolio_values"])
        print(f"✓ Backtest completed: {len(portfolio_values)} periods")
        return results

    def _calculate_metrics(self, portfolio_values: np.ndarray) -> Dict:
        """성과 지표 계산"""
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        n_weeks = len(returns)
        n_years = n_weeks / 52 if n_weeks > 0 else 0.0001

        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        annualized_return = (
            (portfolio_values[-1] / portfolio_values[0]) ** (1 / n_years) - 1
        ) * 100
        annualized_vol = np.std(returns) * np.sqrt(52) * 100
        sharpe_ratio = (
            annualized_return / annualized_vol if annualized_vol > 0 else 0.0
        )

        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdowns) * 100

        win_rate = np.sum(returns > 0) / len(returns) * 100 if len(returns) > 0 else 0.0

        return {
            "Total Return (%)": total_return,
            "Annualized Return (%)": annualized_return,
            "Annualized Volatility (%)": annualized_vol,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown (%)": max_drawdown,
            "Win Rate (%)": win_rate,
        }

    def print_metrics(self, metrics: Dict) -> None:
        """성과 지표 출력"""
        print("\nEnsemble Backtest Performance Metrics")
        print("-" * 40)
        for k, v in metrics.items():
            if "Ratio" in k:
                print(f"{k:30s}: {v:.3f}")
            else:

                print(f"{k:30s}: {v:.2f}")
