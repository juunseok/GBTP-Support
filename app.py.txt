import pandas as pd
import streamlit as st
from urllib.parse import urlparse

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기업 지원사업 검색",
    layout="wide",
)

st.title("기업 지원사업 검색")
st.caption("왼쪽 필터(지역/분야/매출액/고용인원)로 조건에 맞는 사업만 조회합니다.")

DATA_PATH = "260206.csv"

# -----------------------------
# 유틸 함수
# -----------------------------
def ensure_url(value: str) -> str:
    """링크가 스킴(http/https) 없이 들어온 경우 https:// 를 붙여줍니다."""
    if value is None:
        return ""
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return ""
    parsed = urlparse(s)
    if parsed.scheme in ("http", "https"):
        return s
    return "https://" + s.lstrip("/")

def split_regions(region_cell: str) -> list[str]:
    """
    '고령군, 문경시, ...' 처럼 콤마로 묶인 지역 문자열을 리스트로 변환
    """
    if region_cell is None:
        return []
    s = str(region_cell)
    if s.strip() == "" or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """후보 컬럼명 중 실제 df에 존재하는 첫 번째 컬럼명을 반환"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"데이터 파일을 불러오지 못했습니다: {DATA_PATH}\n\n{e}")
    st.stop()

# 컬럼명(사용자 요구와 실제 CSV가 다를 수 있어, 후보군으로 매핑)
COL_TITLE = pick_existing_column(df, ["사업명", "사업", "지원사업명", "프로그램명"])
COL_REGION = pick_existing_column(df, ["지역", "소재지", "권역"])
COL_FIELD = pick_existing_column(df, ["분야", "산업분야", "업종", "산업"])
COL_SALES = pick_existing_column(df, ["매출액", "매출액조건", "매출", "매출액(억원)"])
COL_EMP = pick_existing_column(df, ["고용인력", "고용인원", "고용", "고용인원(명)"])
COL_LINK = pick_existing_column(df, ["링크", "공고링크", "URL", "홈페이지", "공고 URL"])

# 링크 정리
if COL_LINK:
    df[COL_LINK] = df[COL_LINK].apply(ensure_url)

# 지역 리스트 컬럼 생성(필터용)
if COL_REGION:
    df["_지역리스트"] = df[COL_REGION].apply(split_regions)
else:
    df["_지역리스트"] = [[] for _ in range(len(df))]

# -----------------------------
# 사이드바 필터 UI
# -----------------------------
st.sidebar.header("필터")

# (1) 지역(멀티선택)
selected_regions = []
if COL_REGION:
    all_regions = sorted({r for sub in df["_지역리스트"] for r in sub})
    selected_regions = st.sidebar.multiselect(
        "지역(멀티선택)",
        options=all_regions,
        default=[],
        help="여러 지역 선택 가능 (사업의 지역 목록에 하나라도 포함되면 조회됩니다).",
    )
else:
    st.sidebar.info("데이터에 '지역' 컬럼이 없어 지역 필터를 생략합니다.")

# (2) 분야(멀티선택으로 제공: 실제 업무에서 복수 선택이 편함)
selected_fields = []
if COL_FIELD:
    all_fields = sorted(df[COL_FIELD].dropna().astype(str).unique().tolist())
    selected_fields = st.sidebar.multiselect(
        "분야",
        options=all_fields,
        default=[],
    )
else:
    st.sidebar.info("데이터에 '분야/산업분야' 컬럼이 없어 분야 필터를 생략합니다.")

# (3) 매출액 범위(슬라이더)
sales_range = None
if COL_SALES and pd.api.types.is_numeric_dtype(df[COL_SALES]):
    min_sales = int(df[COL_SALES].min())
    max_sales = int(df[COL_SALES].max())
    sales_range = st.sidebar.slider(
        "매출액(억원) 범위",
        min_value=min_sales,
        max_value=max_sales,
        value=(min_sales, max_sales),
        step=1,
    )
else:
    st.sidebar.info("데이터에 숫자형 '매출액' 컬럼이 없어 매출액 필터를 생략합니다.")

# (4) 고용인력 범위(슬라이더)
emp_range = None
if COL_EMP and pd.api.types.is_numeric_dtype(df[COL_EMP]):
    min_emp = int(df[COL_EMP].min())
    max_emp = int(df[COL_EMP].max())
    emp_range = st.sidebar.slider(
        "고용인원(명) 범위",
        min_value=min_emp,
        max_value=max_emp,
        value=(min_emp, max_emp),
        step=1,
    )
else:
    st.sidebar.info("데이터에 숫자형 '고용인원' 컬럼이 없어 고용인원 필터를 생략합니다.")

st.sidebar.divider()

# -----------------------------
# 필터링 로직
# -----------------------------
filtered = df.copy()

# 지역 필터: 선택된 지역 중 하나라도 포함되면 통과
if COL_REGION and selected_regions:
    filtered = filtered[filtered["_지역리스트"].apply(lambda lst: any(r in lst for r in selected_regions))]

# 분야 필터
if COL_FIELD and selected_fields:
    filtered = filtered[filtered[COL_FIELD].astype(str).isin(selected_fields)]

# 매출액 범위
if sales_range and COL_SALES and pd.api.types.is_numeric_dtype(filtered[COL_SALES]):
    lo, hi = sales_range
    filtered = filtered[(filtered[COL_SALES] >= lo) & (filtered[COL_SALES] <= hi)]

# 고용인원 범위
if emp_range and COL_EMP and pd.api.types.is_numeric_dtype(filtered[COL_EMP]):
    lo, hi = emp_range
    filtered = filtered[(filtered[COL_EMP] >= lo) & (filtered[COL_EMP] <= hi)]

# -----------------------------
# 메인 화면 출력
# -----------------------------
# 표시용 컬럼 정리: 내부 컬럼 제거
display_df = filtered.drop(columns=["_지역리스트"], errors="ignore").copy()

if display_df.empty:
    st.warning("조건에 맞는 사업이 없습니다")
else:
    # 링크 컬럼을 클릭 가능하게 표시
    # LinkColumn은 Streamlit 버전에 따라 동작하므로 requirements에 streamlit>=1.29 권장
    if COL_LINK:
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
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

# 참고: 간단 검색(선택사항)
st.divider()
st.subheader("빠른 검색(선택사항)")
keyword = st.text_input("사업명/주관기관 등에서 키워드를 검색합니다.", value="").strip()

if keyword:
    tmp = df.copy()
    # 문자열 컬럼 대상으로만 검색
    text_cols = [c for c in tmp.columns if tmp[c].dtype == "object" and c != "_지역리스트"]
    mask = False
    for c in text_cols:
        mask = mask | tmp[c].astype(str).str.contains(keyword, case=False, na=False)
    result = tmp[mask].drop(columns=["_지역리스트"], errors="ignore")
    if result.empty:
        st.info("키워드에 해당하는 결과가 없습니다.")
    else:
        if COL_LINK:
            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
                column_config={
                    COL_LINK: st.column_config.LinkColumn(
                        "링크",
                        display_text="바로가기",
                    )
                },
            )
        else:
            st.dataframe(result, use_container_width=True, hide_index=True)
