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
                "이름": row["name"],
                "국어": row["kor"],
                "영어": row["eng"],
                "컴퓨터": row["com"],
                "총점": total,
                "평균": avg,
                "학점": get_grade(avg),
            }
        )

    result.sort(key=lambda row: (-row["총점"], row["이름"]))
    previous_total = None
    current_rank = 0
    for index, row in enumerate(result, start=1):
        if row["총점"] != previous_total:
            current_rank = index
            previous_total = row["총점"]
        row["석차"] = current_rank

    result.sort(key=lambda row: (row["석차"], row["이름"]))
    return [
        {
            "석차": row["석차"],
            "이름": row["이름"],
            "국어": row["국어"],
            "영어": row["영어"],
            "컴퓨터": row["컴퓨터"],
            "총점": row["총점"],
            "평균": row["평균"],
            "학점": row["학점"],
        }
        for row in result
    ]


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


def main() -> None:
    st.set_page_config(page_title="성적표 조회", page_icon="📘", layout="wide")
    st.markdown(
        """
        <style>
        .main .block-container { max-width: 1120px; padding-top: 2rem; }
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

    st.subheader("전체 성적 목록")
    st.caption("총점 기준 석차순으로 정렬됩니다. 총점이 같으면 같은 석차로 표시합니다.")
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
