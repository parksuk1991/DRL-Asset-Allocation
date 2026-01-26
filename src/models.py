"""
경로: src/models.py
DRL 모델 정의
Transformer + MLP 기반 특징 추출기 및 정책/가치 네트워크
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer 기반 특징 추출기
    매크로 변수와 공분산 행렬을 별도로 인코딩 후 결합
    """
    
    def __init__(self, 
                 observation_space: spaces.Box,
                 features_dim: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            observation_space: 관측 공간 (33차원)
            features_dim: 출력 특징 차원
            n_heads: Attention head 수
            n_layers: Transformer layer 수
            dropout: Dropout 비율
        """
        super().__init__(observation_space, features_dim)
        
        # 입력 차원 분리
        self.macro_dim = 17  # 매크로 변수
        self.cov_dim = 16    # 공분산 행렬 (4x4 flatten)
        
        # 매크로 변수 임베딩
        self.macro_embedding = nn.Sequential(
            nn.Linear(self.macro_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 공분산 행렬 임베딩 (4x4 구조 활용)
        self.cov_embedding = nn.Sequential(
            nn.Linear(self.cov_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # 최종 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(128, features_dim),  # 64*2 (macro + cov)
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (batch_size, 33) 상태 텐서
            
        Returns:
            features: (batch_size, features_dim) 특징 벡터
        """
        # 매크로와 공분산 분리
        macro = observations[:, :self.macro_dim]
        cov = observations[:, self.macro_dim:]
        
        # 임베딩
        macro_emb = self.macro_embedding(macro)  # (batch, 64)
        cov_emb = self.cov_embedding(cov)        # (batch, 64)
        
        # Transformer 입력: (batch, seq_len=2, d_model=64)
        transformer_input = torch.stack([macro_emb, cov_emb], dim=1)
        
        # Transformer 인코딩
        transformer_output = self.transformer(transformer_input)  # (batch, 2, 64)
        
        # 평균 풀링
        pooled = transformer_output.mean(dim=1)  # (batch, 64)
        
        # 결합 및 최종 출력
        combined = torch.cat([macro_emb, cov_emb], dim=1)  # (batch, 128)
        features = self.output_layer(combined)
        
        return features


class MLPFeatureExtractor(BaseFeaturesExtractor):
    """
    단순 MLP 기반 특징 추출기 (비교 실험용)
    """
    
    def __init__(self, 
                 observation_space: spaces.Box,
                 features_dim: int = 128,
                 hidden_dims: list = [256, 128]):
        """
        Args:
            observation_space: 관측 공간 (33차원)
            features_dim: 출력 특징 차원
            hidden_dims: 은닉층 차원 리스트
        """
        super().__init__(observation_space, features_dim)
        
        layers = []
        input_dim = observation_space.shape[0]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # 최종 출력층
        layers.append(nn.Linear(input_dim, features_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class CustomActorCriticPolicy(nn.Module):
    """
    커스텀 Actor-Critic 정책
    특징 추출기를 공유하고 별도의 정책/가치 헤드 사용
    """
    
    def __init__(self, 
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 features_extractor_class=TransformerFeatureExtractor,
                 features_dim: int = 128):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        
        # 특징 추출기
        self.features_extractor = features_extractor_class(
            observation_space, 
            features_dim
        )
        
        # Actor (정책 네트워크): 평균과 로그 표준편차 출력
        self.actor_mean = nn.Linear(features_dim, action_space.shape[0])
        self.actor_logstd = nn.Parameter(
            torch.zeros(action_space.shape[0])
        )
        
        # Critic (가치 네트워크)
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, obs: torch.Tensor):
        """정책과 가치 계산"""
        features = self.features_extractor(obs)
        
        # 정책 (행동 분포의 평균)
        action_mean = self.actor_mean(features)
        action_mean = torch.tanh(action_mean)  # -1 ~ 1 범위로 제한
        
        # 가치
        value = self.critic(features)
        
        return action_mean, value
    
    def get_distribution(self, obs: torch.Tensor):
        """행동 분포 반환 (PPO용)"""
        features = self.features_extractor(obs)
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(self.actor_logstd)
        
        return action_mean, action_std


def create_policy_kwargs(use_transformer: bool = True,
                         features_dim: int = 128) -> dict:
    """
    Stable Baselines3용 정책 kwargs 생성
    
    Args:
        use_transformer: Transformer 사용 여부
        features_dim: 특징 차원
        
    Returns:
        policy_kwargs: 정책 설정 딕셔너리
    """
    if use_transformer:
        features_extractor_class = TransformerFeatureExtractor
    else:
        features_extractor_class = MLPFeatureExtractor
    
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": [dict(pi=[128, 64], vf=[128, 64])]  # Actor/Critic 별도 네트워크
    }
    
    return policy_kwargs


if __name__ == "__main__":
    # 테스트
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    
    # Transformer 특징 추출기 테스트
    extractor = TransformerFeatureExtractor(obs_space, features_dim=128)
    dummy_obs = torch.randn(8, 33)  # batch_size=8
    features = extractor(dummy_obs)
    print(f"Transformer output shape: {features.shape}")
    
    # MLP 특징 추출기 테스트
    mlp_extractor = MLPFeatureExtractor(obs_space, features_dim=128)
    mlp_features = mlp_extractor(dummy_obs)
    print(f"MLP output shape: {mlp_features.shape}")
    
    # 정책 kwargs 생성
    policy_kwargs = create_policy_kwargs(use_transformer=True)

    print(f"\nPolicy kwargs: {policy_kwargs}")
