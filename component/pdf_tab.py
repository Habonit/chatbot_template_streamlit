import streamlit as st
import pandas as pd
import time
from pathlib import Path


def render_pdf_tab(
    on_upload: callable,
    on_process: callable,
    on_delete: callable,
    chunks: list = None,
) -> None:
    st.header("PDF Preprocessing")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload PDF (max 20MB)",
            type=["pdf"],
            help="텍스트 추출 가능한 PDF만 지원됩니다.",
        )

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"File: {uploaded_file.name} ({file_size_mb:.2f} MB)")

            if file_size_mb > 20:
                st.error("파일 크기가 20MB를 초과합니다.")
            else:
                if st.button("Upload", type="primary"):
                    on_upload(uploaded_file)
                    st.success("파일이 업로드되었습니다.")
                    st.caption("다음 단계: 오른쪽의 'Start Preprocessing' 버튼을 클릭하세요.")

    with col2:
        st.markdown("### Actions")
        if st.button("Start Preprocessing", disabled=not uploaded_file):
            _run_preprocessing(on_process)

        if st.button("Delete Index", type="secondary"):
            on_delete()
            st.info("인덱스가 삭제되었습니다.")

    st.divider()

    if chunks:
        _render_chunk_stats(chunks)
        _render_chunk_table(chunks)
    else:
        _render_empty_state()


def _render_empty_state():
    """chunks가 없을 때 안내 메시지 표시"""
    st.caption(
        "PDF 문서가 아직 전처리되지 않았습니다. "
        "PDF를 업로드하고 전처리를 시작하면 여기에 청크 목록과 통계가 표시됩니다."
    )
    st.markdown(
        "**전처리 과정:** 텍스트 추출 → 청킹 → LLM 정규화 → 임베딩 생성"
    )


def _format_time(seconds: float) -> str:
    """초를 읽기 쉬운 형식으로 변환"""
    if seconds < 60:
        return f"약 {int(seconds)}초"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"약 {minutes}분 {secs}초"


def _run_preprocessing(on_process: callable) -> None:
    progress_bar = st.progress(0)
    status_container = st.container()
    time_container = st.empty()

    steps = [
        (0.2, "텍스트 추출 중..."),
        (0.4, "청킹 중..."),
        (0.6, "정규화 중..."),
        (0.8, "임베딩 생성 중..."),
        (1.0, "완료"),
    ]

    step_times = {}

    for progress, status in steps:
        progress_bar.progress(progress)

        if progress < 1.0:
            with status_container:
                with st.spinner(status):
                    # 청크 수 기반 시간 추정 (정규화 및 임베딩 단계)
                    if status == "정규화 중..." and "chunks" in st.session_state:
                        chunk_count = len(st.session_state.chunks)
                        estimated_seconds = chunk_count * 2  # 청크당 약 2초
                        time_container.caption(
                            f"📊 처리 대상: {chunk_count}개 청크 | "
                            f"예상 소요 시간: {_format_time(estimated_seconds)}"
                        )
                    elif status == "임베딩 생성 중..." and "chunks" in st.session_state:
                        chunk_count = len(st.session_state.chunks)
                        estimated_seconds = chunk_count * 0.5  # 청크당 약 0.5초
                        time_container.caption(
                            f"📊 처리 대상: {chunk_count}개 청크 | "
                            f"예상 소요 시간: {_format_time(estimated_seconds)}"
                        )
                    else:
                        time_container.empty()

                    start_time = time.time()
                    result = on_process(status)
                    elapsed = time.time() - start_time
                    step_times[status] = elapsed

                    if result and result.get("error"):
                        st.error(result["error"])
                        return

    time_container.empty()
    total_time = sum(step_times.values())
    st.success(f"전처리가 완료되었습니다! (총 소요 시간: {_format_time(total_time)})")
    st.caption("Chat 탭에서 PDF 관련 질문을 할 수 있습니다.")


def _render_chunk_stats(chunks: list) -> None:
    st.markdown("### Statistics")

    total_chunks = len(chunks)
    avg_length = sum(len(c.normalized_text) for c in chunks) / total_chunks if chunks else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", total_chunks)
    col2.metric("Avg Length", f"{avg_length:.0f} chars")
    col3.metric("Embedding Dim", "768")


def _render_chunk_table(chunks: list) -> None:
    st.markdown("### Chunk List")

    data = []
    for chunk in chunks:
        data.append({
            "Index": chunk.chunk_index,
            "Original (preview)": chunk.original_text[:100] + "..." if len(chunk.original_text) > 100 else chunk.original_text,
            "Normalized (preview)": chunk.normalized_text[:100] + "..." if len(chunk.normalized_text) > 100 else chunk.normalized_text,
            "Page": chunk.source_page,
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
