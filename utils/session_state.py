"""
Streamlit 세션 상태 관리
"""

import streamlit as st
import tempfile
import os
from pathlib import Path


def init_session_state():
    """세션 상태 초기화"""
    
    # 데이터 관련
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # 모델 관련
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'training_config' not in st.session_state:
        st.session_state.training_config = None
    if 'states' not in st.session_state:
        st.session_state.states = None
    if 'train_indices' not in st.session_state:
        st.session_state.train_indices = None
    if 'valid_indices' not in st.session_state:
        st.session_state.valid_indices = None
    if 'test_indices' not in st.session_state:
        st.session_state.test_indices = None
    
    # 백테스트 관련
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = None
    
    # 국면 분석 관련
    if 'regime_labels' not in st.session_state:
        st.session_state.regime_labels = None
    if 'regime_summary' not in st.session_state:
        st.session_state.regime_summary = None


def save_uploaded_file(uploaded_file):
    """업로드된 파일을 임시 디렉토리에 저장"""
    
    # 임시 디렉토리 생성
    temp_dir = Path(tempfile.gettempdir()) / "drl_asset_allocation"
    temp_dir.mkdir(exist_ok=True)
    
    # 파일 저장
    file_path = temp_dir / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def get_temp_dir():
    """임시 디렉토리 경로 반환"""
    temp_dir = Path(tempfile.gettempdir()) / "drl_asset_allocation"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def clear_session_state():
    """세션 상태 초기화"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
