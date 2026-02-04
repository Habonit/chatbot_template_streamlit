import streamlit as st
import pandas as pd
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


def _run_preprocessing(on_process: callable) -> None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = [
        (0.2, "텍스트 추출 중..."),
        (0.4, "청킹 중..."),
        (0.6, "정규화 중..."),
        (0.8, "임베딩 생성 중..."),
        (1.0, "완료"),
    ]

    for progress, status in steps:
        status_text.text(status)
        progress_bar.progress(progress)

        if progress < 1.0:
            result = on_process(status)
            if result and result.get("error"):
                st.error(result["error"])
                return

    st.success("전처리가 완료되었습니다!")


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
