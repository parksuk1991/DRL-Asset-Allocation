"""
경로: src/environment.py
강화학습 환경 정의 (수정 완료)
Gymnasium 기반 자산배분 환경

수정 사항:
- 보상 함수 개선 (차별화된 신호)
- 파라미터 설명 추가
- HHI 페널티 조정
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict


class AssetAllocationEnv(gym.Env):
    """
    자산배분 강화학습 환경
    
    Reward = Log Return + λ1*Shannon Entropy - λ2*HHI - λ3*Turnover
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 states: np.ndarray,
                 returns: np.ndarray,
                 valid_indices: np.ndarray,
                 risk_aversion: float = 1.0,
                 transaction_cost: float = 0.001,
                 entropy_coef: float = 0.01,
                 hhi_coef: float = 0.01,
                 turnover_coef: float = 0.001):
        """
        Args:
            states: 상태 행렬 (n_samples x 33)
                   경제 지표, 기술적 지표 등을 포함한 특징들
            returns: 수익률 행렬 (n_samples x n_assets)
                    각 자산의 주간 수익률
            valid_indices: 유효한 시점 인덱스 (충분한 히스토리가 있는 시점)
            risk_aversion: 위험 회피 계수 (높을수록 변동성 회피)
            transaction_cost: 거래 비용 (거래량 * 이 값)
            entropy_coef: Shannon Entropy 보상 계수
                         높을수록 분산투자 장려 (하지만 너무 높으면 모두 같은 비중)
            hhi_coef: HHI (Herfindahl Index) 페널티 계수
                     높을수록 집중도 페널티 증가
            turnover_coef: Turnover 페널티 계수
                          높을수록 거래 회피
        """
        super().__init__()
        
        self.states = states
        self.returns = returns
        self.valid_indices = valid_indices
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        self.entropy_coef = entropy_coef
        self.hhi_coef = hhi_coef
        self.turnover_coef = turnover_coef
        
        # 상태 및 행동 공간
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32
        )
        
        # 행동 공간: 4개 자산에 대한 원시 신호 (-1 ~ 1)
        # softmax로 비중으로 변환됨
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # 초기화
        self.current_step = 0
        self.max_steps = len(valid_indices) - 1
        self.current_weights = np.array([0.25, 0.25, 0.25, 0.25])
        self.previous_weights = self.current_weights.copy()
        
        self.portfolio_values = [1.0]
        self.action_history = []
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """환경 초기화"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_weights = np.array([0.25, 0.25, 0.25, 0.25])
        self.previous_weights = self.current_weights.copy()
        self.portfolio_values = [1.0]
        self.action_history = []
        
        state = self.states[self.valid_indices[self.current_step]]
        
        return state.astype(np.float32), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행"""
        # 1. 행동을 포트폴리오 비중으로 변환
        new_weights = self._action_to_weights(action)
        
        # 2. 다음 시점 수익률
        t = self.valid_indices[self.current_step]
        next_returns = self.returns[t + 1] if t + 1 < len(self.returns) else np.zeros(4)
        
        # 3. 포트폴리오 수익률
        portfolio_return = np.dot(new_weights, next_returns)
        log_return = np.log(1 + portfolio_return) if portfolio_return > -1 else -10
        
        # 4. Shannon Entropy Bonus (분산투자 장려)
        epsilon = 1e-10
        shannon_entropy = -np.sum(new_weights * np.log(new_weights + epsilon))
        
        # 5. HHI Concentration Penalty (집중도 페널티)
        # ⚠️ 수정: hhi_coef를 너무 높게 설정하면 모든 비중이 균등해짐
        # HHI = sum(weight^2), 균등 분배 시 0.25, 집중도 높을 때 1.0에 가까움
        hhi = np.sum(new_weights ** 2)
        
        # 6. Turnover Penalty (거래 비용)
        turnover = np.sum(np.abs(new_weights - self.previous_weights))
        trading_cost = turnover * self.transaction_cost
        
        # 7. 통합 보상 함수 (수정: 더 나은 차별화)
        # ⚠️ 핵심: entropy 페널티가 과도하면 균등 비중이 최적이 됨
        reward = (log_return + 
                 self.entropy_coef * shannon_entropy - 
                 self.hhi_coef * (hhi - 0.25) -  # 수정: 균등 분배의 기준점 ---> Universe 숫자가 증가하면 보상함수의 명시적 0.25는 수정해야함 현재는 4종목 뿐이라 1/4 = 0.25인 것
                 self.turnover_coef * turnover)
        
        # 8. 상태 업데이트
        self.previous_weights = self.current_weights.copy()
        self.current_weights = new_weights
        self.current_step += 1
        
        # 포트폴리오 가치 업데이트
        self.portfolio_values.append(
            self.portfolio_values[-1] * (1 + portfolio_return - trading_cost)
        )
        self.action_history.append(new_weights.copy())
        
        # 9. 종료 조건
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # 10. 다음 상태
        if not terminated:
            next_state = self.states[self.valid_indices[self.current_step]]
        else:
            next_state = np.zeros(33)
        
        info = {
            'portfolio_return': portfolio_return,
            'weights': new_weights,
            'trading_cost': trading_cost,
            'portfolio_value': self.portfolio_values[-1],
            'shannon_entropy': shannon_entropy,
            'hhi': hhi,
            'turnover': turnover,
            'log_return': log_return,
        }
        
        return next_state.astype(np.float32), reward, terminated, truncated, info
    
    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        원시 행동을 포트폴리오 비중으로 변환
        Softmax 정규화를 사용하여 항상 0~1 범위의 비중으로 변환
        """
        exp_action = np.exp(action - np.max(action))  # 수치 안정성
        weights = exp_action / exp_action.sum()
        return weights
    
    def get_performance_metrics(self) -> Dict:
        """성과 지표 계산"""
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - 1.0) * 100
        annualized_return = (portfolio_values[-1] ** (52 / len(returns)) - 1) * 100
        volatility = np.std(returns) * np.sqrt(52) * 100
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown) * 100
    
    def render(self):
        if len(self.portfolio_values) > 1:
            print(f"Step: {self.current_step}, "
                  f"Portfolio Value: {self.portfolio_values[-1]:.4f}, "
                  f"Weights: {self.current_weights}")


if __name__ == "__main__":
    # 테스트
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    loader = DataLoader()
    data = loader.get_aligned_data()
    
    engineer = FeatureEngineer()
    states, valid_indices = engineer.create_state_features(
        data['returns'], 
        data['macro']
    )
    
    env = AssetAllocationEnv(states, data['returns'].values, valid_indices)
    
    # 랜덤 에피소드
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"\nTotal Reward: {total_reward:.4f}")
    print(f"Performance: {env.get_performance_metrics()}")

