"""
경로: DRL-Asset-Allocation/pages/1_📊_Training.py
모델 학습 페이지
"""

import streamlit as st
import numpy as np
import pandas as pd
import yaml
import sys
from pathlib import Path
import time

# 프로젝트 루트를 Python 경로에 추가
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.environment import AssetAllocationEnv
from src.models import create_policy_kwargs
from src.agent import DRLAgent
from utils.session_state import init_session_state, save_uploaded_file, get_temp_dir

# 페이지 설정
st.set_page_config(
    page_title="Training - DRL Asset Allocation",
    page_icon="📊",
    layout="wide"
)

# 세션 상태 초기화
init_session_state()

st.title("📊 Model Training")
st.markdown("---")

# 데이터 확인
if not st.session_state.data_loaded or st.session_state.uploaded_data is None:
    st.warning("⚠️ 먼저 Home 페이지에서 데이터를 업로드하세요.")
    st.stop()

st.success("✅ 데이터가 로드되었습니다.")

# 설정 섹션
st.header("⚙️ 학습 설정")

col1, col2 = st.columns(2)

with col1:
    st.subheader("데이터 분할")
    train_ratio = st.slider("훈련 데이터 비율", 0.5, 0.9, 0.7, 0.05)
    valid_ratio = st.slider("검증 데이터 비율", 0.05, 0.3, 0.15, 0.05)
    test_ratio = round(1 - train_ratio - valid_ratio, 2)
    st.info(f"테스트 데이터 비율: {test_ratio:.2f}")
    
    st.subheader("환경 설정")
    risk_aversion = st.number_input("위험 회피 계수", 0.1, 2.0, 0.5, 0.1)
    transaction_cost = st.number_input("거래 비용", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")

with col2:
    st.subheader("모델 설정")
    algorithm = st.selectbox("알고리즘", ["PPO", "A2C", "SAC"])
    use_transformer = st.checkbox("Transformer 사용", value=True)
    features_dim = st.selectbox("특징 차원", [64, 128, 256], index=1)
    
    st.subheader("학습 설정")
    total_timesteps = st.number_input("총 학습 스텝", 10000, 200000, 50000, 10000)
    learning_rate = st.number_input("학습률", 0.00001, 0.01, 0.0003, 0.00001, format="%.5f")

# 학습 시작 버튼
st.markdown("---")

if st.button("🚀 학습 시작", type="primary", use_container_width=True):
    
    # 진행 상태 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. 데이터 준비
        status_text.text("📁 Step 1/4: 데이터 준비 중...")
        progress_bar.progress(10)
        
        # 임시 파일로 저장
        temp_dir = get_temp_dir()
        data_path = temp_dir / "uploaded_data.csv"
        st.session_state.uploaded_data.to_csv(data_path, index=False)
        
        # 데이터 로드
        loader = DataLoader(data_path=str(data_path))
        data = loader.get_aligned_data()
        
        progress_bar.progress(20)
        
        # 특징 엔지니어링
        engineer = FeatureEngineer(rolling_window=52)
        states, valid_indices = engineer.create_state_features(
            data['returns'],
            data['macro'],
            macro_lag=1
        )
        
        progress_bar.progress(30)
        
        # 데이터 분할
        n_valid = len(valid_indices)
        train_size = int(n_valid * train_ratio)
        valid_size = int(n_valid * valid_ratio)
        
        train_indices = valid_indices[:train_size]
        valid_indices_split = valid_indices[train_size:train_size + valid_size]
        test_indices = valid_indices[train_size + valid_size:]
        
        # 정규화
        states, norm_params = engineer.normalize_features(states, train_indices)
        
        st.session_state.processed_data = data
        st.session_state.states = states
        st.session_state.train_indices = train_indices
        st.session_state.valid_indices = valid_indices_split
        st.session_state.test_indices = test_indices
        
        progress_bar.progress(40)
        status_text.text("✅ Step 1/4: 데이터 준비 완료!")
        
        # 2. 환경 생성
        status_text.text("🎮 Step 2/4: 환경 생성 중...")
        
        env = AssetAllocationEnv(
            states=states,
            returns=data['returns'].values,
            valid_indices=train_indices,
            risk_aversion=risk_aversion,
            transaction_cost=transaction_cost,
            entropy_coef=0.01,
            hhi_coef=0.01,
            turnover_coef=0.001,
        )
        
        progress_bar.progress(50)
        status_text.text("✅ Step 2/4: 환경 생성 완료!")
        
        # 3. 에이전트 생성
        status_text.text("🤖 Step 3/4: 에이전트 생성 중...")
        
        policy_kwargs = create_policy_kwargs(
            use_transformer=use_transformer,
            features_dim=features_dim
        )
        
        agent = DRLAgent(
            env=env,
            algorithm=algorithm,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            device='auto',
            seed=42,
        )
        
        progress_bar.progress(60)
        status_text.text("✅ Step 3/4: 에이전트 생성 완료!")
        
        # 4. 학습
        status_text.text("🎓 Step 4/4: 모델 학습 중... (시간이 걸릴 수 있습니다)")
        
        # 학습 시작 시간
        start_time = time.time()
        
        # 학습 콜백 (진행률 업데이트)
        class StreamlitCallback:
            def __init__(self, total_steps, progress_bar, status_text):
                self.total_steps = total_steps
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.last_update = time.time()
                
            def __call__(self, locals_dict, globals_dict):
                current_step = locals_dict.get('self').num_timesteps
                progress = 60 + int(40 * current_step / self.total_steps)
                
                # 1초마다 업데이트
                if time.time() - self.last_update > 1:
                    self.progress_bar.progress(min(progress, 100))
                    self.status_text.text(
                        f"🎓 Step 4/4: 학습 중... {current_step}/{self.total_steps} steps ({progress-60:.0f}%)"
                    )
                    self.last_update = time.time()
                
                return True
        
        # 학습 실행
        agent.train(total_timesteps=total_timesteps)
        
        # 학습 완료
        elapsed = time.time() - start_time
        progress_bar.progress(100)
        status_text.text(f"✅ Step 4/4: 학습 완료! (소요 시간: {elapsed/60:.1f}분)")
        
        # 세션에 저장
        st.session_state.trained_model = agent
        st.session_state.training_config = {
            'algorithm': algorithm,
            'use_transformer': use_transformer,
            'features_dim': features_dim,
            'total_timesteps': total_timesteps,
            'learning_rate': learning_rate,
            'train_ratio': train_ratio,
            'valid_ratio': valid_ratio,
            'test_ratio': test_ratio,
        }
        
        # 성공 메시지
        st.balloons()
        st.success(f"""
        ✅ **학습이 완료되었습니다!**
        
        - 알고리즘: {algorithm}
        - 총 학습 스텝: {total_timesteps:,}
        - 소요 시간: {elapsed/60:.1f}분
        - 훈련 샘플: {len(train_indices)}개
        - 검증 샘플: {len(valid_indices_split)}개
        - 테스트 샘플: {len(test_indices)}개
        """)
        
        st.info("👉 이제 **Backtest** 페이지로 이동하여 성과를 분석하세요.")
        
    except Exception as e:
        st.error(f"❌ 학습 중 오류 발생: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# 학습 이력 표시
st.markdown("---")
st.header("📋 학습 이력")

if st.session_state.trained_model is not None:
    config = st.session_state.training_config
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("알고리즘", config['algorithm'])
        st.metric("학습 스텝", f"{config['total_timesteps']:,}")
    with col2:
        st.metric("모델 타입", "Transformer" if config['use_transformer'] else "MLP")
        st.metric("특징 차원", config['features_dim'])
    with col3:
        st.metric("학습률", f"{config['learning_rate']:.5f}")
        st.metric("훈련/검증/테스트", f"{config['train_ratio']:.0%}/{config['valid_ratio']:.0%}/{config['test_ratio']:.0%}")
    
    # 데이터 정보
    if st.session_state.processed_data is not None:
        st.subheader("데이터 정보")
        data = st.session_state.processed_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**테스트 기간**")
            test_start = data['dates'].iloc[st.session_state.test_indices[0]]
            test_end = data['dates'].iloc[st.session_state.test_indices[-1]]
            st.write(f"{test_start} ~ {test_end}")
        
        with col2:
            st.write("**샘플 수**")
            st.write(f"훈련: {len(st.session_state.train_indices)}")
            st.write(f"검증: {len(st.session_state.valid_indices)}")
            st.write(f"테스트: {len(st.session_state.test_indices)}")

else:
    st.info("아직 학습된 모델이 없습니다.")
