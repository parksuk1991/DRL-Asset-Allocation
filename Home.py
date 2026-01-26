"""
DRL Asset Allocation - Streamlit App
메인 홈페이지
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="DRL Asset Allocation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# 타이틀
st.title("🤖 DRL-Based Asset Allocation System")
st.markdown("---")

# 소개
st.markdown("""
## 📌 프로젝트 소개

이 시스템은 **Deep Reinforcement Learning (DRL)**을 활용한 지능형 자산배분 모델입니다.

### 주요 기능

1. **📊 데이터 준비 및 학습**
   - Bloomberg 데이터 업로드
   - Look-Ahead Bias 방지
   - PPO/A2C/SAC 알고리즘 학습

2. **📈 백테스팅**
   - 테스트 기간 성과 분석
   - S&P500, Equal Weight 벤치마크 비교
   - 인터랙티브 Plotly 차트

3. **🎯 국면 분석**
   - t-SNE + K-Means 기반 국면 발견
   - 국면별 포트폴리오 특성 분석
   - 전환 패턴 시각화

4. **💼 실무 배포**
   - 최신 포트폴리오 비중 산출
   - 리스크 체크 및 신뢰도 평가
   - 모델 재학습 스케줄링

### 시작하기

왼쪽 사이드바에서 원하는 페이지로 이동하세요.

1. **Training**: 데이터 업로드 및 모델 학습
2. **Backtest**: 백테스트 실행 및 성과 분석
3. **Regime Analysis**: 국면 발견 및 시각화
4. **Production**: 실무 배포용 포트폴리오 산출
""")

st.markdown("---")

# 데이터 업로드 섹션
st.header("📁 데이터 업로드")

uploaded_file = st.file_uploader(
    "Bloomberg 데이터 CSV 파일을 업로드하세요",
    type=['csv'],
    help="컬럼 A: 날짜, B~E: 지수 가격, F~V: 매크로 변수"
)

if uploaded_file is not None:
    try:
        # 데이터 로드
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        
        # 기본 검증
        if df.shape[1] < 22:
            st.error(f"❌ 컬럼 수가 부족합니다. 최소 22개 필요 (현재: {df.shape[1]}개)")
        else:
            st.success("✅ 데이터가 성공적으로 로드되었습니다!")
            
            # 데이터 미리보기
            st.subheader("데이터 미리보기")
            st.dataframe(df.head(10), use_container_width=True)
            
            # 기본 통계
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 행 수", f"{len(df):,}")
            with col2:
                st.metric("총 컬럼 수", df.shape[1])
            with col3:
                date_range = f"{df.iloc[0, 0]} ~ {df.iloc[-1, 0]}"
                st.metric("기간", date_range)
            
            # 세션에 저장
            st.session_state.uploaded_data = df
            st.session_state.data_loaded = True
            
            st.info("👉 이제 왼쪽 사이드바에서 **Training** 페이지로 이동하여 모델을 학습하세요.")
            
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {str(e)}")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Powered by Streamlit | Deep Reinforcement Learning Asset Allocation</p>
</div>
""", unsafe_allow_html=True)
