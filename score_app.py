import random
import sqlite3
from pathlib import Path

import streamlit as st


DB_PATH = Path(__file__).with_name("scoreDB.db")

STUDENTS = [
    ("도연", 90, 95, 95),
    ("서준", 94, 88, 91),
    ("하윤", 97, 93, 90),
    ("민서", 89, 96, 94),
    ("지우", 92, 91, 98),
    ("서연", 99, 94, 96),
    ("현우", 88, 90, 93),
    ("지민", 95, 97, 89),
    ("윤아", 91, 92, 99),
    ("준호", 96, 89, 92),
]


def get_grade(avg: float) -> str:
    if avg >= 90:
        return "A"
    if avg >= 80:
        return "B"
    if avg >= 70:
        return "C"
    if avg >= 60:
        return "D"
    return "F"


def connect_db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with connect_db() as conn:
        cur = conn.cursor()
        cur.execute("drop table if exists score")
        cur.execute(
            """
            create table score(
                id integer primary key autoincrement,
                name text not null,
                kor integer not null,
                eng integer not null,
                com integer not null
            )
            """
        )
        cur.executemany(
            "insert into score(name, kor, eng, com) values (?, ?, ?, ?)",
            STUDENTS,
        )
        conn.commit()


def ensure_db() -> None:
    if not DB_PATH.exists():
        init_db()
        return

    with connect_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "select name from sqlite_master where type='table' and name='score'"
        )
        exists = cur.fetchone() is not None

    if not exists:
        init_db()


def read_scores() -> list[dict]:
    ensure_db()
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "select id, name, kor, eng, com from score order by id"
        ).fetchall()

    result = []
    for row in rows:
        total = row["kor"] + row["eng"] + row["com"]
        avg = round(total / 3, 1)
        result.append(
            {
                "번호": row["id"],
                "이름": row["name"],
                "국어": row["kor"],
                "영어": row["eng"],
                "컴퓨터": row["com"],
                "총점": total,
                "평균": avg,
                "학점": get_grade(avg),
            }
        )
    return result


def add_random_student() -> None:
    names = ["가은", "나윤", "다인", "라희", "민재", "수빈", "예준", "유나", "은우", "채원"]
    record = (
        random.choice(names),
        random.randint(88, 99),
        random.randint(88, 99),
        random.randint(88, 99),
    )
    with connect_db() as conn:
        conn.execute(
            "insert into score(name, kor, eng, com) values (?, ?, ?, ?)",
            record,
        )
        conn.commit()


def render_report_card(row: dict) -> None:
    st.markdown(
        f"""
        <div class="report-card">
            <div class="report-title">********* {row['이름']}님의 성적표 *********</div>
            <div class="score-grid">
                <span>이름</span><strong>{row['이름']}</strong>
                <span>국어</span><strong>{row['국어']} 점</strong>
                <span>영어</span><strong>{row['영어']} 점</strong>
                <span>컴퓨터</span><strong>{row['컴퓨터']} 점</strong>
            </div>
            <div class="summary">
                <span>총점 <b>{row['총점']} 점</b></span>
                <span>평균 <b>{row['평균']} 점</b></span>
                <span>학점 <b>{row['학점']} 학점</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="성적표 조회", page_icon="📘", layout="wide")
    st.markdown(
        """
        <style>
        .main .block-container { max-width: 1120px; padding-top: 2rem; }
        .report-card {
            border: 1px solid #d8dee9;
            border-radius: 8px;
            background: #ffffff;
            padding: 28px 32px;
            box-shadow: 0 8px 24px rgba(31, 41, 55, 0.08);
            margin: 18px 0 28px;
        }
        .report-title {
            text-align: center;
            font-size: 26px;
            font-weight: 800;
            color: #172033;
            margin-bottom: 26px;
        }
        .score-grid {
            display: grid;
            grid-template-columns: 120px 1fr;
            gap: 12px 18px;
            max-width: 420px;
            margin: 0 auto 24px;
            font-size: 22px;
        }
        .score-grid span { color: #5a677d; }
        .score-grid strong { color: #172033; }
        .summary {
            display: flex;
            justify-content: center;
            gap: 34px;
            flex-wrap: wrap;
            border-top: 1px solid #e5e7eb;
            padding-top: 20px;
            font-size: 22px;
        }
        .summary b { color: #0f766e; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("성적표 조회")
    st.caption("SQLite score table에서 학생 점수를 읽어 총점, 평균, 학점을 계산합니다.")

    scores = read_scores()

    col1, col2, col3 = st.columns(3)
    col1.metric("학생 수", f"{len(scores)}명")
    col2.metric("전체 평균", f"{round(sum(r['평균'] for r in scores) / len(scores), 1)}점")
    col3.metric("A 학점", f"{sum(1 for r in scores if r['학점'] == 'A')}명")

    st.subheader("학생별 성적표")
    selected_name = st.selectbox("학생 선택", [row["이름"] for row in scores])
    selected = next(row for row in scores if row["이름"] == selected_name)
    render_report_card(selected)

    st.subheader("전체 성적 목록")
    st.dataframe(scores, use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1, 1])
    if c1.button("랜덤 학생 1명 추가"):
        add_random_student()
        st.rerun()
    if c2.button("기본 10명 데이터로 초기화"):
        init_db()
        st.rerun()


if __name__ == "__main__":
    main()
