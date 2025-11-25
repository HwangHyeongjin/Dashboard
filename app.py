import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------
# 1. 공통: CSV 로딩 함수
# ------------------------------
def read_csv_any(path, encodings=("utf-8-sig", "cp949", "euc-kr", "latin1")) -> pd.DataFrame:
    """
    여러 인코딩을 시도해서 CSV를 읽는 함수.
    파일 인코딩이 헷갈릴 때 안전하게 쓰기 위해 사용.
    """
    raw = Path(path).read_bytes()
    for enc in encodings:
        try:
            txt = raw.decode(enc)
            return pd.read_csv(io.StringIO(txt))
        except Exception:
            continue

    # 그래도 안 되면 그냥 latin1 + ignore 로 밀어붙이기
    txt = raw.decode("latin1", errors="ignore")
    return pd.read_csv(io.StringIO(txt))


@st.cache_data
def load_data():
    fire = read_csv_any("화재출동데이터_2024.csv")
    weather = read_csv_any("기상데이터_2024.csv")
    pop = read_csv_any("인구데이터_2024.csv")
    return fire, weather, pop


# ------------------------------
# 2. 데이터 전처리
# ------------------------------
def preprocess_fire(fire: pd.DataFrame):
    """화재 데이터에서 서울만 추출하고, 월별/지역별/원인별 집계 생성"""
    seoul = fire[fire["GRNDS_CTPV_NM"] == "서울특별시"].copy()

    seoul["연도"] = seoul["OCRN_YR"]
    seoul["월"] = seoul["OCRN_MM"]
    seoul["발생일"] = pd.to_datetime(seoul["OCRN_YMD"].astype(str), format="%Y%m%d")

    # 월별 화재 건수
    monthly = seoul.groupby(["연도", "월"]).size().reset_index(name="화재건수")

    # 원인(대분류)별 건수
    cause = (
        seoul.groupby("IGTN_DMNT_LCLSF_NM")
        .size()
        .reset_index(name="건수")
        .sort_values("건수", ascending=False)
    )

    # 자치구별 건수
    region = (
        seoul.groupby("GRNDS_SGG_NM")
        .size()
        .reset_index(name="건수")
        .sort_values("건수", ascending=False)
    )

    return seoul, monthly, cause, region


def preprocess_weather(weather: pd.DataFrame):
    """기상 데이터에서 서울만 추출하고, 월별/일별 집계 생성"""
    w = weather[weather["지점명"] == "서울"].copy()
    w["일시"] = pd.to_datetime(w["일시"])

    w["연도"] = w["일시"].dt.year
    w["월"] = w["일시"].dt.month
    w["날짜"] = w["일시"].dt.date

    # 월별 평균 기온/습도, 월강수량
    monthly = (
        w.groupby(["연도", "월"])
        .agg(
            {
                "기온(°C)": "mean",
                "습도(%)": "mean",
                "강수량(mm)": "sum",
            }
        )
        .reset_index()
    )

    # 일별 평균 기온/습도, 일강수량
    daily = (
        w.groupby("날짜")
        .agg(
            {
                "기온(°C)": "mean",
                "습도(%)": "mean",
                "강수량(mm)": "sum",
            }
        )
        .reset_index()
    )
    daily["날짜"] = pd.to_datetime(daily["날짜"])

    return w, monthly, daily


def make_daily_merge(seoul_fire: pd.DataFrame, daily_weather: pd.DataFrame):
    """일별 화재 건수 + 기상 데이터 결합 (기상-화재 상관분석용)"""
    daily_fire = seoul_fire.groupby("발생일").size().reset_index(name="화재건수")

    merged = pd.merge(
        daily_fire,
        daily_weather,
        left_on="발생일",
        right_on="날짜",
        how="inner",
    )
    return merged


# ------------------------------
# 3. 페이지별 화면 함수
# ------------------------------
def page_overview(seoul_fire, monthly_fire, cause_fire, region_fire):
    st.header("서울시 화재 현황 대시보드")

    # ---- 필터 영역 ----
    col1, col2 = st.columns(2)
    years = sorted(seoul_fire["연도"].unique())
    with col1:
        year = st.selectbox("연도 선택", years, index=len(years) - 1)
    with col2:
        gu_options = ["전체"] + sorted(seoul_fire["GRNDS_SGG_NM"].unique())
        gu = st.selectbox("자치구 선택", gu_options, index=0)

    filtered = seoul_fire[seoul_fire["연도"] == year].copy()
    if gu != "전체":
        filtered = filtered[filtered["GRNDS_SGG_NM"] == gu]

    # ---- KPI 카드 ----
    k1, k2, k3 = st.columns(3)
    k1.metric("화재 건수", len(filtered))
    k2.metric(
        "인명 피해(사상자 수)",
        int(filtered["DTH_CNT"].sum() + filtered["INJPSN_CNT"].sum()),
    )
    k3.metric("재산 피해액(억원)", round(filtered["PRPT_DAM_AMT"].sum() / 1e8, 2))

    # ==============================
    #   월별 화재 / 사망자 추세
    # ==============================
    st.subheader(f"{year}년 월별 화재/사망자 추세")

    c_top1, c_top2 = st.columns(2)

    # 1) 월별 화재 건수 (1~12월 강제 표시)
    monthly_fire_trend = (
        filtered.groupby("월")
        .size()
        .reindex(range(1, 13), fill_value=0)
        .reset_index()
    )
    monthly_fire_trend.columns = ["월", "화재건수"]

    with c_top1:
        fig_fire = px.line(
            monthly_fire_trend,
            x="월",
            y="화재건수",
            markers=True,
            title=f"{year}년 월별 화재 발생 추세",
            color_discrete_sequence=["#A855F7"],  # 보라색
        )
        fig_fire.update_layout(
            xaxis_title="월",
            yaxis_title="건수",
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig_fire, use_container_width=True)

    # 2) 월별 사망자 추세 (1~12월 강제 표시)
    monthly_death_trend = (
        filtered.groupby("월")["DTH_CNT"]
        .sum()
        .reindex(range(1, 13), fill_value=0)
        .reset_index()
    )
    monthly_death_trend.columns = ["월", "사망자수"]

    with c_top2:
        fig_death = px.line(
            monthly_death_trend,
            x="월",
            y="사망자수",
            markers=True,
            title=f"{year}년 월별 사망자 추세",
            color_discrete_sequence=["#A855F7"],  # 보라색
        )
        fig_death.update_layout(
            xaxis_title="월",
            yaxis_title="사망자 수",
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig_death, use_container_width=True)

    # ==============================
    #   하단: 원인 & 자치구 바 차트
    # ==============================
    c1, c2 = st.columns(2)

    with c1:
        cause_sel = (
            filtered.groupby("IGTN_DMNT_LCLSF_NM")
            .size()
            .reset_index(name="건수")
            .sort_values("건수", ascending=False)
            .head(10)
        )
        fig_cause = px.bar(
            cause_sel,
            x="IGTN_DMNT_LCLSF_NM",
            y="건수",
            title="발화 요인(대분류)별 화재 건수 TOP 10",
            color_discrete_sequence=["#A855F7"],  # 보라색
        )
        fig_cause.update_layout(xaxis_title="발화 요인", yaxis_title="건수")
        st.plotly_chart(fig_cause, use_container_width=True)

    with c2:
        region_sel = (
            seoul_fire[seoul_fire["연도"] == year]
            .groupby("GRNDS_SGG_NM")
            .size()
            .reset_index(name="건수")
            .sort_values("건수", ascending=False)
        )
        fig_region = px.bar(
            region_sel,
            x="GRNDS_SGG_NM",
            y="건수",
            title=f"{year}년 자치구별 화재 건수",
            color_discrete_sequence=["#A855F7"],  # 보라색
        )
        fig_region.update_layout(xaxis_title="자치구", yaxis_title="건수")
        st.plotly_chart(fig_region, use_container_width=True)


def page_weather_relation(daily_merge, monthly_merge):
    st.header("기상 요인과 화재 발생 관계 탐색")

    tab1, tab2 = st.tabs(["일별 산점도", "월별 상관관계"])

    # -------- 탭 1: 일별 산점도 --------
    with tab1:
        st.subheader("일별 기상 요인 vs 화재 건수")

        variable = st.selectbox(
            "기상 변수 선택",
            ["기온(°C)", "습도(%)", "강수량(mm)"],
            index=0,
        )

        fig_scatter = px.scatter(
            daily_merge,
            x=variable,
            y="화재건수",
            trendline="ols",
            title=f"일별 {variable} vs 화재 건수",
        )
        # 산점도 색 보라로
        fig_scatter.update_traces(marker=dict(color="#A855F7"))
        fig_scatter.update_layout(xaxis_title=variable, yaxis_title="화재 건수")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # -------- 탭 2: 월별 상관관계 --------
    with tab2:
        st.subheader("월별 평균 기상 요인과 화재 건수")

        fig_line_temp = px.line(
            monthly_merge,
            x="월",
            y="기온(°C)",
            markers=True,
            title="월별 평균 기온",
        )
        fig_line_temp.update_traces(line=dict(color="#A855F7"))
        fig_line_temp.update_layout(xaxis_title="월", yaxis_title="기온(°C)")

        fig_line_fire = px.line(
            monthly_merge,
            x="월",
            y="화재건수",
            markers=True,
            title="월별 화재 건수",
        )
        fig_line_fire.update_traces(line=dict(color="#A855F7"))
        fig_line_fire.update_layout(xaxis_title="월", yaxis_title="화재 건수")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_line_temp, use_container_width=True)
        with c2:
            st.plotly_chart(fig_line_fire, use_container_width=True)

        st.markdown("#### 기상 변수별 상관계수 (월 단위)")
        corr_df = monthly_merge[["화재건수", "기온(°C)", "습도(%)", "강수량(mm)"]].corr()
        st.dataframe(corr_df.style.format("{:.3f}"))


# ------------------------------
# 4. 메인 함수 (네비게이션)
# ------------------------------
def main():
    st.set_page_config(
        page_title="화재 안전 대시보드",
        layout="wide",
    )

    st.sidebar.title("화면 선택")

    fire, weather, pop = load_data()

    # 전처리
    seoul_fire, monthly_fire, cause_fire, region_fire = preprocess_fire(fire)
    w_all, monthly_weather, daily_weather = preprocess_weather(weather)

    # 월별 화재 + 기상 데이터 merge
    monthly_merge = pd.merge(
        monthly_fire,
        monthly_weather,
        on=["연도", "월"],
        how="inner",
    )

    # 일별 화재 + 기상 merge
    daily_merge = make_daily_merge(seoul_fire, daily_weather)

    page = st.sidebar.radio(
        "",
        ("메인 대시보드", "기상-화재 상관분석"),
    )

    if page == "메인 대시보드":
        page_overview(seoul_fire, monthly_fire, cause_fire, region_fire)
    else:
        page_weather_relation(daily_merge, monthly_merge)


if __name__ == "__main__":
    main()