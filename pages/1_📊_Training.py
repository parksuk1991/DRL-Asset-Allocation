"""
경로: DRL-Asset-Allocation/pages/1_📊_Training.py
모델 학습 페이지 (파라미터 설명 추가)
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
    st.warning("⚠️ 먼저 Home 페이지의 왼쪽 사이드바에서 데이터를 업로드하세요.")
    st.stop()

st.success("✅ 데이터가 로드되었습니다.")

# 설정 섹션
st.header("⚙️ 학습 설정")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 데이터 분할 (Data Split)")
    st.markdown("""
    - **훈련 데이터**: 모델이 학습하는 데이터
    - **검증 데이터**: 과적합 방지용
    - **테스트 데이터**: 최종 평가용 (학습에 미사용)
    """)
    
    train_ratio = st.slider(
        "훈련 데이터 비율 (Train Ratio)",
        0.5, 0.9, 0.7, 0.05,
        help="과거 데이터의 몇 %를 학습에 사용할지 결정"
    )
    valid_ratio = st.slider(
        "검증 데이터 비율 (Validation Ratio)",
        0.05, 0.3, 0.15, 0.05,
        help="학습 중 과적합 방지용 데이터 비율"
    )
    test_ratio = round(1 - train_ratio - valid_ratio, 2)
    st.info(f"테스트 데이터 비율: **{test_ratio:.2f}** (자동 계산)")
    
    st.subheader("🎮 환경 설정 (Environment Config)")
    st.markdown("""
    - **위험 회피**: 변동성 회피 정도
    - **거래 비용**: 시장의 거래 비용 반영
    """)
    
    risk_aversion = st.number_input(
        "위험 회피 계수 (Risk Aversion)",
        0.1, 2.0, 0.5, 0.1,
        help="높을수록 변동성 회피, 안정적 포트폴리오"
    )
    transaction_cost = st.number_input(
        "거래 비용 (Transaction Cost)",
        0.0001, 0.01, 0.001, 0.0001,
        format="%.4f",
        help="실제 시장 거래 비용 (0.001 = 0.1%)"
    )

with col2:
    st.subheader("🤖 모델 설정 (Model Config)")
    st.markdown("""
    - **알고리즘**: PPO(권장), A2C, SAC
    - **Transformer**: 시계열 패턴 포착
    - **특징 차원**: 네트워크 크기
    """)
    
    algorithm = st.selectbox(
        "알고리즘 (Algorithm)",
        ["PPO", "A2C", "SAC"],
        help="PPO: 안정적(권장), A2C: 가벼움, SAC: 효율적"
    )
    
    use_transformer = st.checkbox(
        "Transformer 사용 (Use Transformer)",
        value=True,
        help="시계열 시퀀스 학습 - 시간 ���보 중요한 경우 유용"
    )
    
    features_dim = st.selectbox(
        "특징 차원 (Features Dimension)",
        [64, 128, 256],
        index=1,
        help="64: 가벼움, 128: 기본(권장), 256: 무거움"
    )
    
    st.subheader("📚 학습 설정 (Training Config)")
    st.markdown("""
    - **총 스텝**: 더 크다고 항상 좋은 것 아님!
    - **학습률**: 높으면 불안정, 낮으면 느림
    """)
    
    total_timesteps = st.number_input(
        "총 학습 스텝 (Total Timesteps)",
        10000, 200000, 50000, 10000,
        help="50,000 ~ 100,000 권장 (환경과 알고리즘에 따라 다름)"
    )
    
    learning_rate = st.number_input(
        "학습률 (Learning Rate)",
        0.00001, 0.01, 0.0003, 0.00001,
        format="%.5f",
        help="기본값 0.0003 권장 (높으면 불안정, 낮으면 느림)"
    )

# 학습 시작 버튼
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**학습을 시작하면 시간이 소요될 수 있습니다.**")
with col2:
    start_training = st.button("🚀 학습 시작", type="primary", use_container_width=True)

if start_training:
    
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
            entropy_coef=0.005,  # 수정: 너무 높지 않게
            hhi_coef=0.005,      # 수정: 집중도 페널티 조정
            turnover_coef=0.0005, # 거래 비용 반영
        )
        
        progress_bar.progress(50)
        status_text.text("✅ Step 2/4: 환경 생성 완료!")
        
        # 3. 에이전트 생��
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
        
        📊 학습 정보:
        - 알고리즘: {algorithm}
        - 총 학습 스텝: {total_timesteps:,}
        - 소요 시간: {elapsed/60:.1f}분
        - 훈련 샘플: {len(train_indices):,}개
        - 검증 샘플: {len(valid_indices_split):,}개
        - 테스트 샘플: {len(test_indices):,}개
        """)
        
        st.info("👉 다음: **Backtest** 페이지로 이동하여 성과를 분석하세요!")
        
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
        st.metric("데이터 분할", f"{config['train_ratio']:.0%} / {config['valid_ratio']:.0%} / {config['test_ratio']:.0%}")
    
    # 데이터 정보
    if st.session_state.processed_data is not None:
        st.subheader("📊 데이터 정보")
        data = st.session_state.processed_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**테스트 기간**")
            test_start = data['dates'].iloc[st.session_state.test_indices[0]]
            test_end = data['dates'].iloc[st.session_state.test_indices[-1]]
            st.write(f"{test_start} ~ {test_end}")
        
        with col2:
            st.write("**샘플 수**")
            st.write(f"훈련: {len(st.session_state.train_indices):,}")
            st.write(f"검증: {len(st.session_state.valid_indices):,}")
            st.write(f"테스트: {len(st.session_state.test_indices):,}")

else:
    st.info("아직 학습된 모델이 없습니다.")
