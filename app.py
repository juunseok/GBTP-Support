import re
from urllib.parse import urlparse

import pandas as pd
import streamlit as st


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="기업 지원사업 검색", layout="wide")
st.title("기업 지원사업 검색")
st.caption("왼쪽 필터 조건에 맞는 사업만 표로 보여주고, 하단에서 사업을 선택하면 전체 세부내용을 확인할 수 있습니다.")

DATA_PATH = "260206.csv"


# -----------------------------
# 유틸 함수
# -----------------------------
def pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """후보 컬럼명 중 실제 df에 존재하는 첫 컬럼명을 반환"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_url(value: str) -> str:
    """링크가 http/https 없이 들어온 경우 https:// 붙이기"""
    if value is None:
        return ""
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return ""
    parsed = urlparse(s)
    if parsed.scheme in ("http", "https"):
        return s
    return "https://" + s.lstrip("/")


def normalize_region_text(s: str) -> str:
    """구분자/괄호/공백 등 정리"""
    s = str(s)
    s = s.replace("·", ",").replace("/", ",").replace(";", ",").replace("|", ",")
    s = s.replace("\n", " ").replace("\t", " ")
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


GYEONGBUK_WHOLE_KEYWORDS = {
    "경상북도", "경북",
    "경상북도 전체", "경상북도전역",
    "경북 전체", "경북전역", "경북 전역",
    "경상북도 전역",
}


def is_gyeongbuk_whole(region_cell: str) -> bool:
    """지역 값이 '경상북도(전역/전체/경북 등)'이면 True"""
    if region_cell is None:
        return False
    raw = str(region_cell).strip()
    if raw == "" or raw.lower() == "nan":
        return False

    raw = normalize_region_text(raw)
    compact = raw.replace(" ", "")

    for kw in GYEONGBUK_WHOLE_KEYWORDS:
        if kw.replace(" ", "") in compact:
            return True
    return False


def split_regions(region_cell: str) -> list[str]:
    """일반 지역 문자열을 콤마 기준 리스트로 변환 (정제 포함)"""
    if region_cell is None:
        return []
    raw = str(region_cell).strip()
    if raw == "" or raw.lower() == "nan":
        return []

    raw = normalize_region_text(raw)

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    cleaned = []
    for p in parts:
        p = p.replace("경상북도", "").replace("경북", "").strip()
        if p:
            cleaned.append(p)

    # 중복 제거(순서 유지)
    seen = set()
    result = []
    for x in cleaned:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    """숫자형 변환: 콤마/공백/단위 섞여도 최대한 숫자로 변환"""
    cleaned = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("명", "", regex=False)
        .str.replace("억원", "", regex=False)
        .str.replace("억", "", regex=False)
    )
    cleaned = cleaned.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def render_detail_block(title: str, value: str):
    """상세 보기에서 항목은 굵게, 값은 일반 텍스트로 표시"""
    v = "" if value is None else str(value)
    if v.strip() == "" or v.strip().lower() == "nan":
        v = "-"
    st.markdown(f"**{title}**  \n{v}")


# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"데이터 파일을 불러오지 못했습니다: {DATA_PATH}\n\n{e}")
    st.stop()

df["_row_id"] = range(len(df))

# 컬럼 매핑
COL_TITLE = pick_existing_column(df, ["사업명", "사업", "지원사업명", "프로그램명"])
COL_REGION = pick_existing_column(df, ["지역", "소재지", "권역"])
COL_FIELD = pick_existing_column(df, ["분야", "산업분야", "업종", "산업"])
COL_SALES = pick_existing_column(df, ["매출액", "매출액조건", "매출", "매출액(억원)"])
COL_EMP = pick_existing_column(df, ["고용인력", "고용인원", "고용", "고용인원(명)"])
COL_LINK = pick_existing_column(df, ["링크", "공고링크", "URL", "홈페이지", "공고 URL"])
COL_ORG = pick_existing_column(df, ["주관기관", "수행기관", "기관", "운영기관"])

if COL_LINK:
    df[COL_LINK] = df[COL_LINK].apply(ensure_url)

# 지역 정제 + 경북전역 플래그
if COL_REGION:
    df["_is_gyeongbuk_whole"] = df[COL_REGION].apply(is_gyeongbuk_whole)
    df["_region_list"] = df[COL_REGION].apply(split_regions)
else:
    df["_is_gyeongbuk_whole"] = False
    df["_region_list"] = [[] for _ in range(len(df))]

# 매출/고용 숫자화
if COL_SALES:
    df["_sales_num"] = coerce_numeric_series(df[COL_SALES])
else:
    df["_sales_num"] = pd.NA

if COL_EMP:
    df["_emp_num"] = coerce_numeric_series(df[COL_EMP])
else:
    df["_emp_num"] = pd.NA


# -----------------------------
# 사이드바 필터
# -----------------------------
st.sidebar.header("필터")

# (1) 지역: 단일 선택 (멀티 선택 X)
selected_region = None
if COL_REGION:
    region_options = sorted({r for lst in df["_region_list"] for r in lst if r})
    # "전체" 옵션 추가
    selected_region = st.sidebar.selectbox("지역", options=["전체"] + region_options, index=0)
else:
    st.sidebar.info("데이터에 '지역' 컬럼이 없어 지역 필터를 생략합니다.")

# (2) 분야: 멀티선택 유지(원하시면 단일로 바꿀 수도 있습니다)
selected_fields = []
if COL_FIELD:
    field_options = sorted(df[COL_FIELD].dropna().astype(str).unique().tolist())
    selected_fields = st.sidebar.multiselect("분야", options=field_options, default=[])
else:
    st.sidebar.info("데이터에 '분야' 컬럼이 없어 분야 필터를 생략합니다.")

# (3) 매출액
sales_mode = None
sales_range = None
sales_choice = None

if COL_SALES:
    sales_numeric_ratio = df["_sales_num"].notna().mean()
    if sales_numeric_ratio >= 0.6 and df["_sales_num"].notna().any():
        sales_mode = "numeric"
        mn = int(df["_sales_num"].min())
        mx = int(df["_sales_num"].max())
        sales_range = st.sidebar.slider(
            "매출액 범위(숫자)",
            min_value=mn,
            max_value=mx,
            value=(mn, mx),
            step=1,
        )
    else:
        sales_mode = "categorical"
        sales_values = sorted(df[COL_SALES].fillna("").astype(str).unique().tolist())
        sales_choice = st.sidebar.selectbox("매출액조건", options=["전체"] + sales_values, index=0)
else:
    st.sidebar.info("데이터에 '매출액/매출액조건' 컬럼이 없어 매출 필터를 생략합니다.")

# (4) 고용인력
emp_mode = None
emp_range = None
emp_choice = None

if COL_EMP:
    emp_numeric_ratio = df["_emp_num"].notna().mean()
    if emp_numeric_ratio >= 0.6 and df["_emp_num"].notna().any():
        emp_mode = "numeric"
        mn = int(df["_emp_num"].min())
        mx = int(df["_emp_num"].max())
        emp_range = st.sidebar.slider(
            "고용인력 범위(숫자)",
            min_value=mn,
            max_value=mx,
            value=(mn, mx),
            step=1,
        )
    else:
        emp_mode = "categorical"
        emp_values = sorted(df[COL_EMP].fillna("").astype(str).unique().tolist())
        emp_choice = st.sidebar.selectbox("고용인력(조건)", options=["전체"] + emp_values, index=0)
else:
    st.sidebar.info("데이터에 '고용인력' 컬럼이 없어 고용 필터를 생략합니다.")


# -----------------------------
# 필터링
# -----------------------------
filtered = df.copy()

# 지역 필터(단일)
# - 사용자가 '전체'를 고르면 지역 필터 없음
# - 특정 지역을 고르면:
#   (a) '경상북도/경북 전역' 사업은 무조건 통과
#   (b) 아니면 해당 지역을 포함하는 경우만 통과
if COL_REGION and selected_region and selected_region != "전체":
    filtered = filtered[
        (filtered["_is_gyeongbuk_whole"] == True)
        | (filtered["_region_list"].apply(lambda lst: selected_region in lst))
    ]

# 분야
if COL_FIELD and selected_fields:
    filtered = filtered[filtered[COL_FIELD].astype(str).isin(selected_fields)]

# 매출
if COL_SALES:
    if sales_mode == "numeric" and sales_range is not None:
        lo, hi = sales_range
        filtered = filtered[filtered["_sales_num"].notna()]
        filtered = filtered[(filtered["_sales_num"] >= lo) & (filtered["_sales_num"] <= hi)]
    elif sales_mode == "categorical" and sales_choice and sales_choice != "전체":
        filtered = filtered[filtered[COL_SALES].fillna("").astype(str) == str(sales_choice)]

# 고용
if COL_EMP:
    if emp_mode == "numeric" and emp_range is not None:
        lo, hi = emp_range
        filtered = filtered[filtered["_emp_num"].notna()]
        filtered = filtered[(filtered["_emp_num"] >= lo) & (filtered["_emp_num"] <= hi)]
    elif emp_mode == "categorical" and emp_choice and emp_choice != "전체":
        filtered = filtered[filtered[COL_EMP].fillna("").astype(str) == str(emp_choice)]


# -----------------------------
# 결과 테이블
# -----------------------------
internal_cols = {"_row_id", "_region_list", "_is_gyeongbuk_whole", "_sales_num", "_emp_num"}
display_df = filtered[[c for c in filtered.columns if c not in internal_cols]].copy()

if display_df.empty:
    st.warning("조건에 맞는 사업이 없습니다")
    st.stop()

st.subheader(f"검색 결과: {len(display_df):,}건")

if COL_LINK and COL_LINK in display_df.columns:
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            COL_LINK: st.column_config.LinkColumn(
                "링크",
                help="클릭하여 공고 페이지로 이동",
                display_text="바로가기",
            )
        },
    )
else:
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# -----------------------------
# 상세 보기 (드롭다운 + 전체 컬럼을 굵은 항목명으로 출력)
# -----------------------------
st.divider()
st.subheader("사업 상세 보기")

if COL_TITLE is None:
    st.info("사업명 컬럼을 찾지 못해 상세 보기를 표시할 수 없습니다. (CSV 컬럼명을 확인해 주세요)")
    st.stop()

def make_label(row: pd.Series) -> str:
    title = str(row.get(COL_TITLE, "")).strip()
    if COL_ORG and COL_ORG in row.index:
        org = str(row.get(COL_ORG, "")).strip()
        if org and org.lower() != "nan":
            return f"{title} | {org}"
    return title

select_source = filtered.copy()
select_source["_label"] = select_source.apply(make_label, axis=1)

labels = select_source["_label"].astype(str).tolist()
row_ids = select_source["_row_id"].astype(int).tolist()

selected_idx = st.selectbox(
    "사업 선택",
    options=list(range(len(labels))),
    format_func=lambda i: labels[i] if i < len(labels) else "",
)

selected_row_id = row_ids[selected_idx]
selected_full = df[df["_row_id"] == selected_row_id].iloc[0]

# 링크는 상단에 한 번 더
if COL_LINK and COL_LINK in df.columns:
    link = str(selected_full.get(COL_LINK, "")).strip()
    if link:
        st.markdown(f"**공고 링크**  \n[{link}]({link})")

# 모든 컬럼을 항목(굵게) + 값(일반)으로 출력
for c in df.columns:
    if c in internal_cols or c == "_label":
        continue

    v = selected_full.get(c, "")
    if pd.isna(v):
        v = ""

    # 링크 컬럼은 마크다운 링크로 표시
    if COL_LINK and c == COL_LINK:
        url = str(v).strip()
        if url:
            render_detail_block(c, f"[{url}]({url})")
        else:
            render_detail_block(c, "-")
    else:
        render_detail_block(c, str(v))
    st.write("")  # 항목 간 간격
