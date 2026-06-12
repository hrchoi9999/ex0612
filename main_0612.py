import io
import hashlib
import os
import re

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
KEYWORD_COLUMNS = ("시군구", "읍면동", "단지명", "도로명", "건축년도", "번지")


st.set_page_config(page_title="CSV 파일 분석", page_icon="📊", layout="wide")
st.title("CSV 파일 분석")
st.write("----------------")


with st.expander("OpenAI API Key 발급 안내", expanded=True):
    st.markdown(
        """
        1. OpenAI Platform에 로그인합니다.
        2. API Keys 페이지에서 **Create new secret key**를 눌러 새 키를 발급합니다.
        3. 발급된 키를 아래 입력창에 붙여 넣습니다.
        4. API 사용량 과금을 위해 결제 수단과 사용 한도를 확인합니다.
        """
    )
    col_api, col_docs, col_billing = st.columns(3)
    with col_api:
        st.link_button("API Key 발급", "https://platform.openai.com/api-keys")
    with col_docs:
        st.link_button("OpenAI Quickstart", "https://developers.openai.com/api/docs/quickstart")
    with col_billing:
        st.link_button("Billing 확인", "https://platform.openai.com/settings/organization/billing/overview")


openai_key = st.text_input("OPENAI_API_KEY", type="password")
uploaded_files = st.file_uploader(
    "분석할 CSV 파일을 하나 이상 올려주세요.",
    type=["csv"],
    accept_multiple_files=True,
)
st.write("----------------")


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def find_csv_header_line(file_bytes, encoding):
    """안내문이 포함된 CSV에서 실제 컬럼 헤더가 시작되는 줄을 찾습니다."""
    decoded = file_bytes.decode(encoding)
    header_keywords = ("NO", "시군구", "단지명", "도로명")
    for index, line in enumerate(decoded.splitlines()):
        if "," in line and any(keyword in line for keyword in header_keywords):
            return index
    return 0


def read_csv_uploaded_file(uploaded_file):
    """여러 한국어 CSV 인코딩과 안내문 헤더를 처리해 DataFrame으로 읽습니다."""
    file_bytes = uploaded_file.getvalue()
    last_error = None

    for encoding in CSV_ENCODINGS:
        try:
            header_line = find_csv_header_line(file_bytes, encoding)
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                skiprows=header_line,
                dtype=str,
                keep_default_na=False,
            )
            df.columns = [str(column).strip() for column in df.columns]
            df = df.loc[df.apply(lambda row: any(clean_value(value) for value in row), axis=1)]
            return df, encoding, header_line
        except Exception as error:
            last_error = error

    raise ValueError(f"CSV 파일을 읽을 수 없습니다. 마지막 오류: {last_error}")


def format_csv_value(column, value):
    text = clean_value(value)
    if not text or text == "-":
        return text

    if "건축년도" in column and text.isdigit():
        return f"{text}년"
    return text


def csv_row_to_document(row, row_index, source_name):
    """CSV 한 행을 ChromaDB 검색에 유리한 텍스트 청크로 변환합니다."""
    fields = []
    keywords = []

    for column, value in row.items():
        column_name = str(column)
        formatted_value = format_csv_value(column_name, value)
        if not formatted_value:
            continue

        fields.append(f"{column_name}: {formatted_value}")

        if column_name in KEYWORD_COLUMNS:
            keywords.append(formatted_value)

    page_content = "\n".join(
        [
            f"파일명: {source_name}",
            f"CSV 행 번호: {row_index + 1}",
            "검색 키워드: " + " ".join(dict.fromkeys(keywords)),
            *fields,
        ]
    )

    return Document(
        page_content=page_content,
        metadata={
            "source": source_name,
            "row_index": row_index + 1,
            "file_type": "csv",
        },
    )


def csv_to_documents(uploaded_file):
    df, encoding, skipped_rows = read_csv_uploaded_file(uploaded_file)
    documents = [
        csv_row_to_document(row, index, uploaded_file.name)
        for index, row in df.iterrows()
    ]
    return documents, df, encoding, skipped_rows


def build_multi_csv_documents(uploaded_files):
    all_documents = []
    csv_infos = []
    preview_frames = []

    for uploaded_file in uploaded_files:
        documents, df, encoding, skipped_rows = csv_to_documents(uploaded_file)
        all_documents.extend(documents)

        preview_df = df.copy()
        preview_df.insert(0, "CSV 행 번호", range(1, len(preview_df) + 1))
        preview_df.insert(0, "파일명", uploaded_file.name)
        preview_frames.append(preview_df)

        csv_infos.append(
            {
                "filename": uploaded_file.name,
                "rows": len(df),
                "columns": len(df.columns),
                "encoding": encoding,
                "skipped_rows": skipped_rows,
            }
        )

    combined_preview = (
        pd.concat(preview_frames, ignore_index=True, sort=False)
        if preview_frames
        else pd.DataFrame()
    )
    return all_documents, csv_infos, combined_preview


def get_upload_signature(uploaded_files):
    signature = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()
        signature.append((uploaded_file.name, len(file_bytes), digest))
    return tuple(signature)


def normalize_for_match(value):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", clean_value(value)).lower()


def extract_match_terms(question, combined_preview):
    compact_question = normalize_for_match(question)
    terms = []

    year_month_match = re.search(r"(20\d{2})\s*년\s*0?(\d{1,2})\s*월", question)
    if year_month_match:
        year, month = year_month_match.groups()
        terms.append(f"{year}{int(month):02d}")

    stopwords = {
        "거래정보",
        "조회",
        "정보",
        "알려주세요",
        "있는",
        "아파트",
        "아파트의",
        "주소",
    }
    row_texts = combined_preview.apply(
        lambda row: normalize_for_match(" ".join(clean_value(value) for value in row)),
        axis=1,
    )

    for token in re.findall(r"[0-9A-Za-z가-힣]+", question):
        token = token.strip()
        compact_token = normalize_for_match(token)
        if len(compact_token) < 2 or compact_token in stopwords:
            continue
        if compact_token in compact_question and row_texts.str.contains(compact_token, regex=False).any():
            terms.append(compact_token)

    return list(dict.fromkeys(terms))


def find_matching_rows(question, combined_preview):
    if combined_preview.empty:
        return pd.DataFrame(), []

    terms = extract_match_terms(question, combined_preview)
    if not terms:
        return pd.DataFrame(), []

    row_texts = combined_preview.apply(
        lambda row: normalize_for_match(" ".join(clean_value(value) for value in row)),
        axis=1,
    )
    mask = pd.Series(True, index=combined_preview.index)
    for term in terms:
        mask &= row_texts.str.contains(term, regex=False)

    return combined_preview.loc[mask].copy(), terms


def matching_rows_to_documents(matching_rows):
    documents = []
    for _, row in matching_rows.iterrows():
        source_name = clean_value(row.get("파일명", "CSV"))
        row_index_text = clean_value(row.get("CSV 행 번호", ""))
        row_index = int(row_index_text) if row_index_text.isdigit() else 0
        row_data = row.drop(labels=["파일명", "CSV 행 번호"], errors="ignore")
        documents.append(csv_row_to_document(row_data, row_index - 1, source_name))
    return documents


def merge_documents(primary_documents, secondary_documents):
    merged = []
    seen = set()

    for document in [*primary_documents, *secondary_documents]:
        key = (
            document.metadata.get("source"),
            document.metadata.get("row_index"),
            document.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(document)

    return merged


class StreamHandler(BaseCallbackHandler):
    """LLM이 생성하는 토큰을 Streamlit 화면에 실시간 출력합니다."""

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)


def show_csv_preview(csv_infos, combined_preview):
    total_rows = sum(info["rows"] for info in csv_infos)
    st.success(f"CSV 로드 완료: {len(csv_infos):,}개 파일, 총 {total_rows:,}행")

    summary_rows = [
        {
            "파일명": info["filename"],
            "행 수": info["rows"],
            "열 수": info["columns"],
            "인코딩": info["encoding"],
            "건너뛴 안내 행": info["skipped_rows"],
        }
        for info in csv_infos
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with st.expander("통합 미리보기", expanded=True):
        st.dataframe(combined_preview.head(50), use_container_width=True)


def show_retrieval_debug(db):
    with st.expander("CSV 검색 품질 확인", expanded=True):
        st.caption(
            "업로드한 모든 CSV 파일의 행 청크를 하나의 ChromaDB에 넣고 유사도 검색합니다."
        )
        default_query = "명륜2가 아남1 아파트의 건축년도와 도로명을 알려줘"
        debug_query = st.text_input("검색 테스트 문장", value=default_query)

        if st.button("관련 행 검색 테스트"):
            results = db.similarity_search_with_score(debug_query, k=5)
            if not results:
                st.warning("검색 결과가 없습니다.")
                return

            for rank, (document, score) in enumerate(results, start=1):
                contains_required_terms = all(
                    term in document.page_content
                    for term in ("아남1", "명륜2가", "건축년도: 1995년", "도로명: 창경궁로 265")
                )
                status = "포함" if contains_required_terms else "확인 필요"
                st.markdown(
                    f"**{rank}. {document.metadata.get('source', '-')} "
                    f"| row {document.metadata.get('row_index', '-')} "
                    f"| distance {score:.4f} | 핵심 청크: {status}**"
                )
                st.code(document.page_content, language="text")


if uploaded_files:
    if not openai_key:
        st.warning("CSV 분석을 시작하려면 OpenAI API Key를 입력해 주세요.")
        st.stop()

    upload_signature = get_upload_signature(uploaded_files)
    cached_store = st.session_state.get("csv_vector_store")

    if cached_store and cached_store["signature"] == upload_signature:
        documents = cached_store["documents"]
        csv_infos = cached_store["csv_infos"]
        combined_preview = cached_store["combined_preview"]
        db = cached_store["db"]
        st.info("업로드한 CSV 파일이 변경되지 않아 기존 벡터 DB를 재사용합니다.")
    else:
        with st.spinner("CSV 파일들을 읽고 통합 벡터 DB를 생성하는 중입니다...", show_time=True):
            documents, csv_infos, combined_preview = build_multi_csv_documents(uploaded_files)
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_key)
            db = Chroma.from_documents(documents=documents, embedding=embeddings)

        st.session_state["csv_vector_store"] = {
            "signature": upload_signature,
            "documents": documents,
            "csv_infos": csv_infos,
            "combined_preview": combined_preview,
            "db": db,
        }

    show_csv_preview(csv_infos, combined_preview)
    show_retrieval_debug(db)

    retriever = db.as_retriever(search_kwargs={"k": 7})

    st.header("업로드한 CSV 전체 내용에 대해 질문하세요")
    question = st.text_input(
        "질문 입력",
        placeholder="예: 명륜2가 아남1의 건축년도와 도로명은?",
    )

    if st.button("질문하기"):
        if not question.strip():
            st.warning("질문을 입력해 주세요.")
        else:
            with st.spinner("답변 생성 중입니다...", show_time=True):
                matching_rows, match_terms = find_matching_rows(question, combined_preview)
                if not matching_rows.empty:
                    st.subheader("조건에 맞는 CSV 원본 행")
                    st.caption(
                        f"질문에서 찾은 조건: {', '.join(match_terms)} | "
                        f"원본 CSV 매칭 행: {len(matching_rows):,}건"
                    )
                    st.dataframe(matching_rows, use_container_width=True, hide_index=True)

                llm = ChatOpenAI(
                    model=CHAT_MODEL,
                    temperature=0,
                    api_key=openai_key,
                )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 여러 CSV 데이터를 함께 분석하는 AI입니다.
                    아래 Context에 있는 내용만 근거로 답변하세요.
                    답변에는 근거가 된 파일명, 행 번호, 컬럼명과 값을 함께 설명하세요.
                    정확한 근거가 Context에 없으면 모른다고 답하세요.

                    Context:
                    {context}

                    Question:
                    {input}

                    답변:
                    """
                )

                document_chain = create_stuff_documents_chain(llm, prompt)
                retrieved_documents = retriever.invoke(question)
                exact_documents = matching_rows_to_documents(matching_rows)
                context_documents = merge_documents(exact_documents, retrieved_documents)
                answer = document_chain.invoke(
                    {
                        "input": question,
                        "context": context_documents,
                    }
                )

                if answer:
                    st.markdown(answer)
                else:
                    st.warning("답변이 비어 있습니다. 질문을 조금 더 구체적으로 입력해 주세요.")

                with st.expander("답변에 사용된 검색 근거", expanded=False):
                    for index, document in enumerate(context_documents, start=1):
                        st.markdown(
                            f"**{index}. {document.metadata.get('source', '-')} "
                            f"| row {document.metadata.get('row_index', '-')}**"
                        )
                        st.code(document.page_content, language="text")
