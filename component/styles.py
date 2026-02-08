"""교육용 챗봇 커스텀 CSS 스타일"""
import streamlit as st


def get_custom_css() -> str:
    """커스텀 CSS 문자열 반환"""
    return """
<style>
/* === 모드 뱃지 스타일 === */
.mode-badge-casual {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    background-color: #E8F5E9;
    color: #2E7D32;
    font-size: 0.85em;
    font-weight: 600;
}
.mode-badge-normal {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    background-color: #E3F2FD;
    color: #1565C0;
    font-size: 0.85em;
    font-weight: 600;
}
.mode-badge-reasoning {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    background-color: #F3E5F5;
    color: #7B1FA2;
    font-size: 0.85em;
    font-weight: 600;
}

/* === 메타데이터 패널 === */
.metadata-panel {
    background-color: #F7F8FA;
    border-left: 3px solid #4A90D9;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.9em;
}

/* === 컨셉 카드 === */
.concept-card {
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    transition: box-shadow 0.2s ease;
}
.concept-card:hover {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* === 사이드바 섹션 간격 === */
section[data-testid="stSidebar"] .stExpander {
    margin-bottom: 4px;
}

/* === 토큰 프로그레스 바 === */
.token-progress {
    margin-top: 4px;
}

/* === Welcome 카드 === */
.welcome-card {
    background: linear-gradient(135deg, #F7F8FA 0%, #E3F2FD 100%);
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    text-align: center;
}

/* === 그래프 경로 표시 === */
.graph-path {
    font-family: monospace;
    font-size: 0.85em;
    color: #546E7A;
}
.graph-path-arrow {
    color: #90A4AE;
    margin: 0 4px;
}
</style>
"""


def inject_custom_css():
    """Streamlit 앱에 커스텀 CSS 주입"""
    css = get_custom_css()
    st.markdown(css, unsafe_allow_html=True)
    return css
