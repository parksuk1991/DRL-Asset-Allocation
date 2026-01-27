"""
경로: src/agent.py
DRL Agent Implementation (수정 완료)

PPO, A2C, SAC 알고리즘 구현
각 파라미터에 대한 상세한 설명 추가
"""

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Dict
import os


class TrainingCallback(BaseCallback):
    """Training progress monitoring callback"""

    def __init__(self, check_freq: int = 1000, save_path: str = "./models/"):
        super().__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                print(f" Step: {self.n_calls}, Mean Reward: {mean_reward:.4f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    save_file = os.path.join(self.save_path, "best_model.zip")
                    self.model.save(save_file)
        return True


class DRLAgent:
    """DRL Agent Wrapper (PPO, A2C, SAC)"""

    def __init__(
        self,
        env,
        algorithm: str = "PPO",
        policy_kwargs: Optional[Dict] = None, 
        learning_rate: float = 3e-4,
        device: str = "auto",
        seed: int = 42,
    ):
        """
        강화학습 에이전트 초기화
        
        Args:
            env: 강화학습 환경 (AssetAllocationEnv)
            algorithm: 사용할 알고리즘 ("PPO", "A2C", "SAC")
            policy_kwargs: 정책 네트워크 구성 (신경망 구조 등)
            learning_rate: 학습률 (높을수록 빠르게 학습하지만 불안정할 수 있음)
            device: 계산 장치 ("cpu", "cuda", "auto")
            seed: 난수 시드 (재현성 보장)
        """
        self.env = DummyVecEnv([lambda: env])
        self.algorithm = algorithm
        self.seed = seed

        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # SAC는 다른 policy kwargs 필요 (off-policy 알고리즘)
        if algorithm in ["SAC"]:
            sac_policy_kwargs = {
                "net_arch": [256, 256]
            }
            policy_kwargs_to_use = sac_policy_kwargs
        else:
            policy_kwargs_to_use = policy_kwargs

        # Select algorithm
        if algorithm == "PPO":
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=learning_rate,
                # n_steps: 정책 업데이트 전에 수집할 경험 수 (높을수록 안정적이지만 느림)
                n_steps=2048,
                # batch_size: 한 번에 처리할 샘플 수 (높을수록 메모리 많이 사용)
                batch_size=64,
                # n_epochs: 수집한 데이터를 몇 번 반복 학습할지
                n_epochs=10,
                # gamma: 할인 인자 (미래 보상의 중요도, 0.99가 기본값)
                gamma=0.99,
                # gae_lambda: 일반화된 이득 추정(GAE) 파라미터 (0.95가 표준)
                gae_lambda=0.95,
                # clip_range: 정책 업데이트 범위 제한 (0.2가 표준)
                clip_range=0.2,
                # ent_coef: 엔트로피 계수 (탐험 수준, 낮을수록 탐험 감��)
                ent_coef=0.01,  # 수정: 0.05 → 0.01 (과도한 탐험 방지)
                # vf_coef: 가치 함수 손실 가중치
                vf_coef=0.5,
                # max_grad_norm: 그래디언트 클리핑 (학습 안정성)
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs_to_use,
                verbose=0,
                device=self.device,
                tensorboard_log="./tensorboard/",
                seed=self.seed,
            )
        elif algorithm == "A2C":
            self.model = A2C(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=learning_rate,
                # n_steps: A2C에서는 정책 업데이트 전 경험 수 (PPO보다 작음)
                # ⚠️ 중요: 이 값을 늘려도 결과가 같은 이유는 아래 참고
                n_steps=512,  # 수정: 256 → 512 (경험 수 증대)
                # gamma: 할인 인자
                gamma=0.99,
                # gae_lambda: A2C도 GAE 사용 가능
                gae_lambda=0.95,
                # ent_coef: A2C의 핵심 문제! 이 값이 높으면 비중이 균등해짐
                # ⚠️ 수정: 0.1 → 0.001 (큰 폭 감소!)
                # 엔트로피 페널티가 너무 높으면 모든 포트폴리오가 균등 비중이 됨
                ent_coef=0.001,
                # vf_coef: 가치 함수 손실 가중치
                vf_coef=0.5,
                # use_rms_prop: RMSprop 옵티마이저 사용 여부
                use_rms_prop=True,
                # max_grad_norm: 그래디언트 클리핑
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs_to_use,
                verbose=0,
                device=self.device,
                tensorboard_log="./tensorboard/",
                seed=self.seed,
            )
        elif algorithm == "SAC":
            self.model = SAC(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=learning_rate,
                # buffer_size: 경험 재생 버퍼 크기 (SAC는 off-policy)
                buffer_size=50000,
                # batch_size: 샘플링할 배치 크기
                batch_size=128,
                # tau: 대상 네트워크 업데이트 속도 (낮을수록 천천히)
                tau=0.005,
                # gamma: 할인 인자
                gamma=0.99,
                # train_freq: 환경 스텝마다 몇 번 학습할지
                train_freq=1,
                # gradient_steps: 매 train_freq 스텝마다 학습 반복 수
                gradient_steps=1,
                policy_kwargs=policy_kwargs_to_use,
                verbose=0,
                device=self.device,
                tensorboard_log="./tensorboard/",
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, total_timesteps: int = 100000,
              callback: Optional[BaseCallback] = None) -> None:
        """
        에이전트 학습
        
        Args:
            total_timesteps: 총 학습 스텝 수
            callback: 학습 중 호출될 콜백 함수
        """
        if callback is None:
            callback = TrainingCallback(check_freq=1000)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )

    def save(self, path: str = "./models/final_model.zip") -> None:
        """
        학습된 모델 저장
        
        Args:
            path: 저장할 경로
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        학습된 모델 불러오기
        
        Args:
            path: 모델 경로
        """
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env, device=self.device)
        elif self.algorithm == "A2C":
            self.model = A2C.load(path, env=self.env, device=self.device)
        elif self.algorithm == "SAC":
            self.model = SAC.load(path, env=self.env, device=self.device)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """
        관찰에 기반한 행동 예측
        
        Args:
            observation: 현재 상태
            deterministic: 결정론적 정책 사용 여부
            
        Returns:
            action: 예측된 행동
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
