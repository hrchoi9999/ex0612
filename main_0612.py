import io
import os
import tempfile

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


st.set_page_config(page_title="PDF/CSV 분석 AI", page_icon="📄", layout="wide")
st.title("📄 PDF/CSV File Reader")
st.write("----------------")


with st.expander("🔑 OpenAI API Key 발급 안내", expanded=True):
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
uploaded_file = st.file_uploader("PDF 또는 CSV 파일을 올려주세요.", type=["pdf", "csv"])
st.write("----------------")


def pdf_to_documents(uploaded_file):
    """Streamlit에 업로드된 PDF를 LangChain Document 목록으로 변환합니다."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_filepath)
        return loader.load()


def find_csv_header_line(file_bytes, encoding):
    """안내문이 포함된 CSV에서 실제 컬럼 헤더가 시작되는 줄을 찾습니다."""
    decoded = file_bytes.decode(encoding)
    for index, line in enumerate(decoded.splitlines()):
        if "," in line and any(keyword in line for keyword in ("NO", "시군구", "단지명", "도로명")):
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
            df = df.loc[df.apply(lambda row: any(clean_value(value) for value in row), axis=1)]
            df.columns = [str(column).strip() for column in df.columns]
            return df, encoding, header_line
        except Exception as error:
            last_error = error

    raise ValueError(f"CSV 파일을 읽을 수 없습니다. 마지막 오류: {last_error}")


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def format_csv_value(column, value):
    text = clean_value(value)
    if not text or text == "-":
        return text

    if "건축년도" in column and text.isdigit():
        return f"{text}년"
    return text


def csv_row_to_document(row, row_index, source_name):
    """CSV 한 행을 검색 친화적인 텍스트 청크로 변환합니다."""
    fields = []
    keywords = []

    for column, value in row.items():
        formatted_value = format_csv_value(str(column), value)
        if not formatted_value:
            continue

        fields.append(f"{column}: {formatted_value}")

        if column in ("시군구", "읍면동", "단지명", "도로명", "건축년도", "번지"):
            keywords.append(formatted_value)

    page_content = "\n".join(
        [
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


def build_documents(uploaded_file):
    extension = uploaded_file.name.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        pages = pdf_to_documents(uploaded_file)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        return text_splitter.split_documents(pages), "pdf", None

    if extension == "csv":
        documents, df, encoding, skipped_rows = csv_to_documents(uploaded_file)
        return documents, "csv", {
            "dataframe": df,
            "encoding": encoding,
            "skipped_rows": skipped_rows,
        }

    raise ValueError("지원하지 않는 파일 형식입니다.")


class StreamHandler(BaseCallbackHandler):
    """LLM이 생성하는 토큰을 Streamlit 화면에 실시간 출력합니다."""

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)


def show_csv_preview(csv_info):
    df = csv_info["dataframe"]
    st.success(
        f"CSV 로드 완료: {len(df):,}행, {len(df.columns):,}열 "
        f"(인코딩: {csv_info['encoding']}, 건너뛴 안내 행: {csv_info['skipped_rows']})"
    )
    st.dataframe(df.head(20), use_container_width=True)


def show_retrieval_debug(db):
    with st.expander("🔎 CSV 검색 품질 확인", expanded=True):
        st.caption(
            "ChromaDB 유사도 검색이 자연어 질문에서 관련 행 청크를 찾는지 확인합니다."
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
                st.markdown(
                    f"**{rank}. row {document.metadata.get('row_index', '-')} "
                    f"| distance {score:.4f} "
                    f"| 핵심 청크 포함: {'✅' if contains_required_terms else '⚠️'}**"
                )
                st.code(document.page_content, language="text")


if uploaded_file is not None:
    if not openai_key:
        st.warning("파일 분석을 시작하려면 OpenAI API Key를 입력해 주세요.")
        st.stop()

    with st.spinner("문서를 읽고 벡터 DB를 생성하는 중입니다...", show_time=True):
        texts, file_type, csv_info = build_documents(uploaded_file)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_key)
        db = Chroma.from_documents(documents=texts, embedding=embeddings)

    if file_type == "csv":
        show_csv_preview(csv_info)
        show_retrieval_debug(db)
    else:
        st.success(f"PDF 로드 완료: 검색 청크 {len(texts):,}개")

    retriever = db.as_retriever(search_kwargs={"k": 5})

    st.header("파일 내용에 대해 질문하세요")
    question = st.text_input(
        "질문 입력",
        placeholder="예: 명륜2가 아남1의 건축년도와 도로명은?",
    )

    if st.button("질문하기"):
        if not question.strip():
            st.warning("질문을 입력해 주세요.")
        else:
            with st.spinner("답변 생성 중입니다...", show_time=True):
                chat_box = st.empty()
                handler = StreamHandler(chat_box)

                llm = ChatOpenAI(
                    model=CHAT_MODEL,
                    temperature=0,
                    api_key=openai_key,
                    streaming=True,
                    callbacks=[handler],
                )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF와 CSV 데이터를 분석하는 AI입니다.
                    아래 Context에 있는 내용만 근거로 답변하세요.
                    CSV 데이터라면 관련 행의 컬럼명과 값을 함께 설명하세요.
                    정확한 근거가 Context에 없으면 모른다고 답하세요.

                    Context:
                    {context}

                    Question:
                    {input}

                    답변:
                    """
                )

                document_chain = create_stuff_documents_chain(llm, prompt)
                qa_chain = create_retrieval_chain(retriever, document_chain)
                qa_chain.invoke({"input": question})
