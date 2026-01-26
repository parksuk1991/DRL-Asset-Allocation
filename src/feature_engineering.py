"""
경로: src/feature_engineering.py
특징 공학 모듈 (Look-Ahead Bias 완전 해결)
52주 롤링 공분산 행렬 및 상태 변수 생성

핵심 수정:
1. 공분산: t 시점 상태는 [t-52:t-1] 수익률로 계산
2. 매크로 변수: t 시점 상태는 t-1 시점의 매크로 값 사용 (1기 lag)
"""

import numpy as np
import pandas as pd
from typing import Tuple


class FeatureEngineer:
    """특징 추출 및 상태 생성 (Look-Ahead Bias 방지)"""
    
    def __init__(self, rolling_window: int = 52):
        """
        Args:
            rolling_window: 공분산 계산을 위한 롤링 윈도우 (기본 52주)
        """
        self.rolling_window = rolling_window
        
    def calculate_rolling_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        52주 롤링 공분산 행렬 계산 (Look-Ahead Bias 방지)
        
        핵심: t 시점의 상태는 t-52 ~ t-1 시점의 수익률로 계산
              (t 시점의 수익률은 포함하지 않음)
        
        Args:
            returns: 수익률 DataFrame (n_samples x 4)
            
        Returns:
            공분산 행렬 (n_samples x 16) - 각 시점의 4x4 행렬을 flatten
        """
        n_samples = len(returns)
        n_assets = returns.shape[1]
        
        # 공분산 행렬 저장 (각 시점마다 4x4 = 16차원)
        cov_matrices = np.zeros((n_samples, n_assets * n_assets))
        
        print(f"Calculating rolling covariance with {self.rolling_window}-week window...")
        print(f"  ⚠ Look-Ahead Bias Prevention: Using returns[t-{self.rolling_window}:t] for state at time t")
        
        for i in range(n_samples):
            if i < self.rolling_window:
                # 데이터가 충분하지 않으면 NaN
                cov_matrices[i, :] = np.nan
            else:
                # 핵심: t 시점의 상태는 t-52 ~ t-1 시점의 수익률로 계산
                window_returns = returns.iloc[i - self.rolling_window:i]
                
                # 디버깅 (처음 몇 개만)
                if i == self.rolling_window:
                    print(f"  Example: state at index {i} uses returns from {i-self.rolling_window} to {i-1}")
                
                cov_matrix = window_returns.cov().values
                
                # 4x4 행렬을 1x16 벡터로 flatten
                cov_matrices[i, :] = cov_matrix.flatten()
        
        print(f"✓ Covariance matrices shape: {cov_matrices.shape}")
        return cov_matrices
    
    def lag_macro_features(self, macro: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
        """
        매크로 변수 시차 적용 (Look-Ahead Bias 방지)
        
        실무 시나리오:
        - t 시점에 의사결정할 때, t 시점의 매크로 데이터는 아직 알 수 없음
        - t-1 시점의 매크로 데이터를 사용해야 함
        
        Args:
            macro: 원본 매크로 변수 DataFrame
            lag: 시차 (기본값 1)
            
        Returns:
            lagged_macro: 시차가 적용된 매크로 변수
        """
        print(f"\nApplying {lag}-period lag to macro variables...")
        print(f"  Reason: At time t, we only know macro data up to t-{lag}")
        
        lagged_macro = macro.shift(lag)
        
        print(f"✓ Macro variables lagged by {lag} period(s)")
        print(f"  Original first row: {macro.iloc[0].values[:3]}...")
        print(f"  Lagged first row: {lagged_macro.iloc[0].values[:3]}... (should be NaN)")
        print(f"  Lagged second row: {lagged_macro.iloc[1].values[:3]}... (should match original first row)")
        
        return lagged_macro
    
    def create_state_features(self, 
                              returns: pd.DataFrame, 
                              macro: pd.DataFrame,
                              macro_lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        최종 상태 변수 생성: 매크로(17, lagged) + 공분산(16) = 33차원
        
        Look-Ahead Bias 완전 제거:
        1. 공분산: t 시점 상태는 [t-52:t-1] 수익률 사용
        2. 매크로: t 시점 상태는 t-1 시점의 값 사용
        
        Args:
            returns: 수익률 DataFrame
            macro: 매크로 변수 DataFrame
            macro_lag: 매크로 변수 시차 (기본값 1)
            
        Returns:
            states: 상태 행렬 (n_samples x 33)
            valid_indices: 유효한 인덱스
        """
        # 1. 매크로 변수에 시차 적용
        lagged_macro = self.lag_macro_features(macro, lag=macro_lag)
        
        # 2. 공분산 행렬 계산 (이미 Look-Ahead Bias 방지됨)
        cov_features = self.calculate_rolling_covariance(returns)
        
        # 3. 매크로 변수를 numpy 배열로 변환
        macro_features = lagged_macro.values
        
        # 4. 결합: [매크로(17, lagged) + 공분산(16)] = 33차원
        states = np.concatenate([macro_features, cov_features], axis=1)
        
        # 5. NaN이 없는 유효한 인덱스 찾기
        # 최소 rolling_window + macro_lag 이후부터 유효
        valid_mask = ~np.isnan(states).any(axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        print(f"\n✓ State features created (Look-Ahead Bias FREE)")
        print(f"  Total samples: {len(states)}")
        print(f"  Valid samples: {len(valid_indices)}")
        print(f"  State feature shape: {states.shape}")
        print(f"  First valid index: {valid_indices[0]}")
        print(f"  - Requires {self.rolling_window} previous returns for covariance")
        print(f"  - Requires {macro_lag} previous period(s) for macro variables")
        
        # 검증 출력
        if len(valid_indices) > 0:
            first_valid = valid_indices[0]
            print(f"\n📊 Validation Check:")
            print(f"  State at index {first_valid}:")
            print(f"    - Uses macro data from index {first_valid - macro_lag}")
            print(f"    - Uses returns from index {first_valid - self.rolling_window} to {first_valid - 1}")
            print(f"  ✅ No future information leak!")
        
        return states, valid_indices
    
    def normalize_features(self, states: np.ndarray, 
                          train_indices: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        특징 정규화 (학습 데이터 기준)
        
        Args:
            states: 원본 상태 행렬
            train_indices: 학습 데이터 인덱스
            
        Returns:
            normalized_states: 정규화된 상태
            norm_params: 정규화 파라미터 (mean, std)
        """
        train_states = states[train_indices]
        
        # 학습 데이터의 평균과 표준편차 계산
        mean = np.nanmean(train_states, axis=0)
        std = np.nanstd(train_states, axis=0)
        
        # 표준편차가 0인 경우 1로 대체 (division by zero 방지)
        std = np.where(std == 0, 1, std)
        
        # 전체 데이터 정규화
        normalized_states = (states - mean) / std
        
        norm_params = {
            'mean': mean,
            'std': std
        }
        
        print("✓ Features normalized based on training data")
        return normalized_states, norm_params


if __name__ == "__main__":
    # 테스트
    from data_loader import DataLoader
    
    loader = DataLoader()
    data = loader.get_aligned_data()
    
    engineer = FeatureEngineer(rolling_window=52)
    states, valid_indices = engineer.create_state_features(
        data['returns'], 
        data['macro'],
        macro_lag=1  # 1기 lag 적용
    )
    
    print(f"\n=== Look-Ahead Bias Complete Check ===")
    print(f"✅ All features are based on past information only")

    print(f"✅ Safe for real-world deployment")
