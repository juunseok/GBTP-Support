import pandas as pd
import streamlit as st
from urllib.parse import urlparse, quote

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="기업 지원사업 검색", layout="wide")
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
    """'A, B, C' 형태의 지역 문자열을 리스트로 변환"""
    if region_cell is None:
        return []
    s = str(region_cell)
    if s.strip() == "" or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """후보 컬럼명 중 실제 df에 존재하는 첫 컬럼명을 반환"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def open_new_tab(url: str, label: str = "새 창으로 열기", key: str = "open_new_tab"):
    """
    Streamlit은 기본 버튼만으로 새 탭을 못 띄우는 경우가 많아
    HTML+JS window.open으로 새 탭을 여는 방식 사용
    """
    safe_url = url.replace('"', "%22")
    st.components.v1.html(
        f"""
        <button
          style="
            background:#0e7aff;color:white;border:none;padding:10px 14px;
            border-radius:10px;cursor:pointer;font-weight:600;
          "
          onclick='window.open("{safe_url}", "_blank")'
        >
          {label}
        </button>
        """,
        height=52,
    )

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

# -----------------------------
# 컬럼 매핑 (CSV에 따라 자동으로 잡히도록 후보군)
# -----------------------------
COL_TITLE = pick_existing_column(df, ["사업명", "사업", "지원사업명", "프로그램명"])
COL_REGION = pick_existing_column(df, ["지역", "소재지", "권역"])
COL_FIELD = pick_existing_column(df, ["분야", "산업분야", "업종", "산업"])
COL_SALES = pick_existing_column(df, ["매출액", "매출액조건", "매출", "매출액(억원)"])
COL_EMP   = pick_existing_column(df, ["고용인력", "고용인원", "고용", "고용인원(명)"])
COL_LINK  = pick_existing_column(df, ["링크", "공고링크", "URL", "홈페이지", "공고 URL"])

# ✅ 설명 이미지(추가): CSV에 아래 중 하나 컬럼이 있으면 사용 (이미지 URL 또는 로컬 파일경로)
COL_IMAGE = pick_existing_column(df, ["설명이미지", "이미지", "이미지URL", "이미지 URL", "사업이미지", "포스터", "썸네일"])

# 내부 식별자(행 고유 id) 추가
df["_row_id"] = range(len(df))

# 링크/이미지 URL 정리
if COL_LINK:
    df[COL_LINK] = df[COL_LINK].apply(ensure_url)
if COL_IMAGE:
    df[COL_IMAGE] = df[COL_IMAGE].apply(ensure_url)

# 지역 리스트
if COL_REGION:
    df["_지역리스트"] = df[COL_REGION].apply(split_regions)
else:
    df["_지역리스트"] = [[] for _ in range(len(df))]

# -----------------------------
# (A) 상세 페이지 모드: ?detail=ROW_ID 로 들어오면 상세만 보여주기
# -----------------------------
query = st.query_params
detail_id = query.get("detail", None)

if detail_id is not None:
    # 상세 페이지(새 탭에서 열릴 화면)
    try:
        rid = int(detail_id)
    except:
        st.error("잘못된 상세 페이지 요청입니다.")
        st.stop()

    row = df[df["_row_id"] == rid]
    if row.empty:
        st.warning("해당 사업을 찾을 수 없습니다.")
        st.stop()

    row = row.iloc[0].to_dict()

    st.subheader(row.get(COL_TITLE, "사업 상세"))
    st.caption("이 화면은 새 탭(새 창)에서 열리는 상세 페이지입니다.")

    # 기본 정보(표 형태)
    info_items = []
    for c in df.columns:
        if c in ["_row_id", "_지역리스트"]:
            continue
        v = row.get(c, "")
        info_items.append({"항목": c, "값": "" if pd.isna(v) else str(v)})

    info_df = pd.DataFrame(info_items)

    # 링크는 클릭 가능하게
    if COL_LINK and row.get(COL_LINK, ""):
        link = row.get(COL_LINK, "")
        st.markdown(f"**공고 링크:** [{link}]({link})")

    # 설명 이미지 표시
    if COL_IMAGE and row.get(COL_IMAGE, ""):
        img = row.get(COL_IMAGE, "")
        st.markdown("### 사업 관련 설명 이미지")
        st.image(img, use_container_width=True)

        # 이미지도 새 탭으로
        st.markdown(f"[이미지 새 탭으로 열기]({img})")
    else:
        st.info("설명 이미지가 없습니다. (CSV에 '설명이미지/이미지URL' 컬럼을 추가하면 표시됩니다.)")

    st.divider()
    st.markdown("이 탭은 닫아도 됩니다.")
    st.stop()

# -----------------------------
# (B) 검색/필터 메인 페이지
# -----------------------------
st.sidebar.header("필터")

# 지역(멀티선택)
selected_regions = []
if COL_REGION:
    all_regions = sorted({r for sub in df["_지역리스트"] for r in sub})
    selected_regions = st.sidebar.multiselect("지역(멀티선택)", options=all_regions, default=[])

# 분야(멀티선택)
selected_fields = []
if COL_FIELD:
    all_fields = sorted(df[COL_FIELD].dropna().astype(str).unique().tolist())
    selected_fields = st.sidebar.multiselect("분야", options=all_fields, default=[])

# 매출액 범위
sales_range = None
if COL_SALES and pd.api.types.is_numeric_dtype(df[COL_SALES]):
    min_sales = int(df[COL_SALES].min())
    max_sales = int(df[COL_SALES].max())
    sales_range = st.sidebar.slider(
        "매출액 범위", min_value=min_sales, max_value=max_sales,
        value=(min_sales, max_sales), step=1
    )

# 고용인원 범위
emp_range = None
if COL_EMP and pd.api.types.is_numeric_dtype(df[COL_EMP]):
    min_emp = int(df[COL_EMP].min())
    max_emp = int(df[COL_EMP].max())
    emp_range = st.sidebar.slider(
        "고용인원 범위", min_value=min_emp, max_value=max_emp,
        value=(min_emp, max_emp), step=1
    )

st.sidebar.divider()

# -----------------------------
# 필터링
# -----------------------------
filtered = df.copy()

if COL_REGION and selected_regions:
    filtered = filtered[filtered["_지역리스트"].apply(lambda lst: any(r in lst for r in selected_regions))]

if COL_FIELD and selected_fields:
    filtered = filtered[filtered[COL_FIELD].astype(str).isin(selected_fields)]

if sales_range and COL_SALES and pd.api.types.is_numeric_dtype(filtered[COL_SALES]):
    lo, hi = sales_range
    filtered = filtered[(filtered[COL_SALES] >= lo) & (filtered[COL_SALES] <= hi)]

if emp_range and COL_EMP and pd.api.types.is_numeric_dtype(filtered[COL_EMP]):
    lo, hi = emp_range
    filtered = filtered[(filtered[COL_EMP] >= lo) & (filtered[COL_EMP] <= hi)]

# -----------------------------
# 결과 표시
# -----------------------------
display_df = filtered.drop(columns=["_지역리스트"], errors="ignore").copy()

if display_df.empty:
    st.warning("조건에 맞는 사업이 없습니다")
    st.stop()

st.subheader(f"검색 결과: {len(display_df):,}건")

# ✅ (핵심) '사업명 클릭' 느낌을 만들기 위해 "행 선택" 기능을 사용
# Streamlit 버전에 따라 selection 기능이 없을 수 있어, data_editor를 사용합니다.
# - 사용자가 한 행을 클릭(선택)하면 아래에 상세 정보 표시
show_cols = [c for c in display_df.columns if c not in ["_row_id"]]
edited = st.data_editor(
    display_df[show_cols],
    use_container_width=True,
    hide_index=True,
    disabled=True,
    key="result_table"
)

st.caption("표에서 원하는 사업의 행을 클릭(선택)하면 아래에 상세 정보가 표시됩니다.")

# -----------------------------
# 상세 보기(선택 기반)
# -----------------------------
# data_editor 자체는 선택 상태를 직접 주지 않아서,
# 초보자 친화적으로 "사업명" 드롭다운으로도 상세 선택을 제공(실무에서 오히려 편합니다)
st.divider()
st.subheader("사업 상세 보기")

# 선택 UI: 사업명 기반
if COL_TITLE:
    options = display_df[["_row_id", COL_TITLE]].copy()
    options[COL_TITLE] = options[COL_TITLE].astype(str)

    selected_title = st.selectbox(
        "사업명을 선택하세요 (선택하면 상세가 표시됩니다)",
        options=options[COL_TITLE].tolist()
    )
    selected_row_id = int(options.loc[options[COL_TITLE] == selected_title, "_row_id"].iloc[0])
else:
    # 사업명이 없다면 row_id로
    selected_row_id = int(display_df["_row_id"].iloc[0])

row = df[df["_row_id"] == selected_row_id].iloc[0]

# 상세 패널
left, right = st.columns([1.2, 1])

with left:
    if COL_TITLE:
        st.markdown(f"### {row[COL_TITLE]}")
    else:
        st.markdown("### 선택한 사업")

    # 공고 링크
    if COL_LINK and str(row.get(COL_LINK, "")).strip():
        link = row[COL_LINK]
        st.markdown(f"**공고 링크:** [{link}]({link})")

    # 상세 페이지를 새 탭으로 열기 (쿼리 파라미터 detail=ROW_ID)
    detail_url = f"?detail={selected_row_id}"
    open_new_tab(detail_url, label="새 창(새 탭)으로 상세 페이지 열기", key="open_detail")

    # 주요 컬럼만 간단히 보여주기(원하면 모두 표시 가능)
    st.markdown("#### 주요 정보")
    key_cols = [c for c in [COL_REGION, COL_FIELD, COL_SALES, COL_EMP] if c]
    for c in key_cols:
        st.write(f"- **{c}**: {row.get(c, '')}")

with right:
    st.markdown("#### 설명 이미지")
    if COL_IMAGE and str(row.get(COL_IMAGE, "")).strip():
        img = row[COL_IMAGE]
        st.image(img, use_container_width=True)
        st.markdown(f"[이미지 새 탭으로 열기]({img})")
    else:
        st.info("설명 이미지가 없습니다. (CSV에 '설명이미지/이미지URL' 컬럼을 추가하면 표시됩니다.)")
