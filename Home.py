"""
경로: Home.py
DRL Asset Allocation - 메인 페이지 (수정 완료)

주요 수정:
1. 데이터 업로드를 사이드바 상단에 배치
2. MIT 라이센스 탭 추가
3. 파라미터 설명 개선
4. 사용자 경험 개선
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
root_path = Path(__file__).parent
sys.path.append(str(root_path))

from utils.session_state import init_session_state, save_uploaded_file, get_temp_dir

# 페이지 설정
st.set_page_config(
    page_title="DRL Asset Allocation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
init_session_state()

# ============================================================================
# 🔴 사이드바: 데이터 업로드 (최상단 배치!)
# ============================================================================
st.sidebar.markdown("## 📁 **데이터 업로드**")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📊 Bloomberg CSV 파일 선택",
    type=['csv'],
    help="""
    다음 컬럼을 포함해야 합니다:
    - Date: 날짜 (YYYY-MM-DD)
    - 자산 수익률: S&P500, KOSPI200, Nikkei225, EuroStoxx50
    - 거시경제지표: (선택사항)
    """
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_data = df
        st.session_state.data_loaded = True
        
        st.sidebar.success("✅ 데이터 업로드 완료!")
        st.sidebar.info(f"""
        📊 데이터 정보:
        - 샘플: {len(df):,}개
        - 컬럼: {df.shape[1]}개
        """)
        
    except Exception as e:
        st.sidebar.error(f"❌ 데이터 로드 실패: {str(e)}")
        st.session_state.data_loaded = False

st.sidebar.markdown("---")

# ============================================================================
# 사이드바: 라이센스
# ============================================================================
with st.sidebar.expander("📜 **MIT License**", expanded=False):
    st.markdown("""
    ```
    MIT License
    
    Copyright (c) 2026 parksuk1991
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    ```
    """)

st.sidebar.markdown("---")

# ============================================================================
# 메인 콘텐츠
# ============================================================================
st.title("🤖 DRL Asset Allocation")
st.markdown("""
**Deep Reinforcement Learning을 이용한 자동 자산배분 시스���**

금리 환경, 시장 상황에 따라 포트폴리오 비중을 자동으로 최적화합니다.
""")

st.markdown("---")

# 데이터 상태 표시
if st.session_state.data_loaded and st.session_state.uploaded_data is not None:
    st.success("✅ 데이터가 준비되었습니다. 왼쪽 사이드바에서 Training을 선택하세요.")
else:
    st.warning("⚠️ 왼쪽 사이드바에서 Bloomberg CSV 파일을 업로드하세요.")

st.markdown("---")

# 소개
st.header("📌 시스템 소개")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎓 **학습**
    - PPO/A2C/SAC 알고리즘
    - 자동 특징 엔지니어링
    - Look-Ahead Bias 제거
    """)

with col2:
    st.markdown("""
    ### 📊 **백테스트**
    - 역사적 성과 분석
    - 벤치마크 비교
    - 상세 시각화
    """)

with col3:
    st.markdown("""
    ### 🎯 **배포**
    - 실시간 포트폴리오
    - 국면 분석
    - 성과 모니터링
    """)

st.markdown("---")

st.header("⚙️ 핵심 파라미터 설명")

with st.expander("🔹 **데이터 분할 (Data Split)**", expanded=True):
    st.markdown("""
    **훈련 데이터 (Train)**: 70%
    - 모델이 학습하는 데이터
    - 과거 충분한 기간의 데이터 필요
    
    **검증 데이터 (Validation)**: 15%
    - 학습 중 과적합(overfitting) 방지
    - 하이퍼파라미터 튜닝용
    
    **테스트 데이터 (Test)**: 15%
    - 최종 성과 평가용
    - 절대 학습에 사용하지 않음 (Look-Ahead Bias 방지)
    """)

with st.expander("🔹 **알고리즘 선택 (Algorithm)**", expanded=True):
    st.markdown("""
    **PPO (Proximal Policy Optimization)**
    - 👍 안정적이고 범용적
    - 👍 학습이 빠름
    - 💡 초보자 추천
    
    **A2C (Advantage Actor-Critic)**
    - 👍 가벼움 (계산량 적음)
    - ⚠️ `ent_coef` 값이 중요! (높으면 균등 비중)
    - 💡 수정됨: ent_coef = 0.001 (기존 0.1 → 0.001)
    
    **SAC (Soft Actor-Critic)**
    - 👍 off-policy로 샘플 효율 높음
    - 👍 탐험과 착취의 균형 자동 조정
    - ⚠️ 계산량이 가장 많음
    """)

with st.expander("🔹 **모델 설정 (Model Config)**", expanded=True):
    st.markdown("""
    **Transformer 사용 (Use Transformer)**
    - ✅ True (권장): 시계열 패턴 포착
    - ❌ False: MLP 네트워크만 사용 (빠르지만 성능 낮음)
    
    **특징 차원 (Features Dimension)**
    - 64: 가볍지만 표현력 낮음
    - 128: 기본값 (권장)
    - 256: 무거움, 과적합 위험
    """)

with st.expander("🔹 **학습 설정 (Training Config)**", expanded=True):
    st.markdown("""
    **총 학습 스텝 (Total Timesteps)**
    - 최소: 10,000 (빠른 테스트용)
    - 권장: 50,000 ~ 100,000
    - 최대: 200,000 (충분한 학습)
    - ℹ️ 더 크다고 더 좋은 �� 아님! 환경 설정도 중요
    
    **학습률 (Learning Rate)**
    - 높음 (0.001): 빠르지만 불안정
    - 기본 (0.0003): 안정적
    - 낮음 (0.00001): 느리지만 섬세
    """)

with st.expander("🔹 **환경 설정 (Environment Config)**", expanded=True):
    st.markdown("""
    **위험 회피 계수 (Risk Aversion)**
    - 높음 (1.0): 변동성 회피, 안정적 포트폴리오
    - 낮음 (0.1): 공격적, 고수익 추구
    
    **거래 비용 (Transaction Cost)**
    - 시장 현실: 0.001 ~ 0.005 (0.1% ~ 0.5%)
    - 높으면: 리밸런싱 빈도 ↓
    - 낮으면: 자주 거래하지만 비용 ↓
    """)

with st.expander("🔹 **백테스트 설정 (Backtest Config)**", expanded=True):
    st.markdown("""
    **리밸런싱 주기 (Rebalance Frequency)**
    - 1주: 매주 조정 (거래 비용 높음)
    - 4주: 월간 조정 (권장)
    - 12주: 분기별 조정 (거래 비용 낮음)
    
    **Trust Region**
    - 0.05: 보수적, 점진적 변화
    - 0.15: 균형잡힌 조정
    - 0.3: 공격적, 빠른 변화
    
    **최소/최대 비중 (Min/Max Weight)**
    - 최소 5%: 과도한 분산 방지
    - 최대 35%: 과도한 집중 방지
    """)

st.markdown("---")

st.header("🚀 시작하기")

st.info("""
1. **📁 데이터 업로드** (사이드바 상단)
   - Bloomberg CSV 파일 선택
   
2. **📊 Training 탭**으로 이동
   - 파라미터 설정
   - 모델 학습 시작
   
3. **📈 Backtest 탭**에서 결과 분석
   - 성과 지표 확인
   - 차트 시각화
   
4. **🎯 Regime Analysis 탭** (선택)
   - 시장 국면 분석
   
5. **💼 Production 탭**
   - 최신 포트폴리오 비중 산출
""")

st.markdown("---")

st.header("💡 주요 개선사항 (v2.0)")

st.success("""
✅ **A2C 모델 수정**
- `ent_coef`: 0.1 → 0.001 (균등 비중 문제 해결)
- `n_steps`: 256 → 512 (경험 수 증대)

✅ **UI/UX 개선**
- 데이터 업로드를 사이드바로 이동
- MIT 라이센스 탭 추가
- 파라미터 설명 추가 (초보자 친화적)

✅ **보상 함수 개선**
- HHI 페널티 조정 (균등 분배 기준점)
- 더 나은 차별화된 신호

✅ **문서화**
- 각 파라미터 상세 설명
- 권장값 제시
""")

st.markdown("---")

st.header("📚 참고 자료")

st.markdown("""
- **DRL Algorithms**: [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- **강화학습 입문**: [Sutton & Barto - Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)
- **포트폴리오 최적화**: [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
""")

st.markdown("---")

st.text("© 2026 parksuk1991 - DRL Asset Allocation System v2.0")
