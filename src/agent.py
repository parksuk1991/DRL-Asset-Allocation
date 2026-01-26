"""
DRL Agent Implementation

PPO, A2C, SAC 알고리즘 구현
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
        Args:
            env: Reinforcement learning environment
            algorithm: Algorithm ("PPO", "A2C", or "SAC")
            policy_kwargs: Policy network configuration
            learning_rate: Learning rate
            device: Device ("cpu", "cuda", "auto")
            seed: Random seed for reproducibility
        """
        self.env = DummyVecEnv([lambda: env])
        self.algorithm = algorithm
        self.seed = seed

        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # SAC는 다른 policy kwargs 필요 (off-policy)
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
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,
                vf_coef=0.5,
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
                n_steps=5,
                gamma=0.99,
                ent_coef=0.05,
                vf_coef=0.5,
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
                buffer_size=50000,
                batch_size=128,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
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
        """Train agent"""
        if callback is None:
            callback = TrainingCallback(check_freq=1000)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
  
        )

    def save(self, path: str = "./models/final_model.zip") -> None:
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load model"""
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env, device=self.device)
        elif self.algorithm == "A2C":
            self.model = A2C.load(path, env=self.env, device=self.device)
        elif self.algorithm == "SAC":
            self.model = SAC.load(path, env=self.env, device=self.device)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Predict action"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
