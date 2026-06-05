"""
==============================================================================
 프로젝트 1-1 — AI 기반 유체물성 예측 모델  (제출용 단독 실행 앱)
==============================================================================
 과목   : 유체역학 (하호진 교수님)  |  학번 202112853  김유민
 -----------------------------------------------------------------------------
 실행 :  streamlit run app_ch1.py
 -----------------------------------------------------------------------------
 [원본 대비 보완 사항]
   (1) 보고서 결론에서 "반드시 수반되어야 한다"고 명시한 외삽(extrapolation)
       예외처리 로직을 predict_safe() 로 실제 구현 → 학습범위 이탈 시 경고.
   (2) 인터페이스 루브릭(A: '단위 선택 가능') 충족 → 온도/물성 단위 변환 제공.
   (3) 예측값 vs NIST 실측값 vs 오차(%) 비교 패널 + R²·RMSE 표시.
   (4) 데이터 파이프라인·모델 하이퍼파라미터를 헤드리스 학습 스크립트
       (ch1_train_export.py) 와 100% 동일하게 유지 → 보고서 수치와 일치.
==============================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 0. 전역 정의
# ─────────────────────────────────────────────────────────────────────────────
PROPS = ["Density_kg_m3", "Viscosity_mPa_s", "Surface_Tension_mN_m", "Vapor_Pressure_kPa"]
NAME_KO = {"Density_kg_m3": "밀도", "Viscosity_mPa_s": "점성계수",
           "Surface_Tension_mN_m": "표면장력", "Vapor_Pressure_kPa": "증기압"}
MODELS_KO = {"RandomForest": "Random Forest", "Polynomial3": "3차 다항회귀"}
TRAIN_LO, TRAIN_HI = 0.0, 100.0           # 학습 온도 범위 (외삽 가드 기준)

# 물성별 단위 변환 테이블 (기준단위 → 표시단위 배율/오프셋)
PROP_UNITS = {
    "Density_kg_m3":        {"kg/m³": (1.0, 0.0), "g/cm³": (1e-3, 0.0)},
    "Viscosity_mPa_s":      {"mPa·s (cP)": (1.0, 0.0), "Pa·s": (1e-3, 0.0)},
    "Surface_Tension_mN_m": {"mN/m": (1.0, 0.0), "N/m": (1e-3, 0.0)},
    "Vapor_Pressure_kPa":   {"kPa": (1.0, 0.0), "Pa": (1e3, 0.0), "bar": (1e-2, 0.0)},
}

def C_to(unit, T_c):
    return T_c if unit == "°C" else (T_c * 9/5 + 32 if unit == "°F" else T_c + 273.15)

def to_C(unit, T):
    return T if unit == "°C" else ((T - 32) * 5/9 if unit == "°F" else T - 273.15)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 파이프라인 (헤드리스 스크립트와 동일 — np.random.seed(42))
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_dataset():
    np.random.seed(42)
    n = 200
    T = np.linspace(0, 100, n)
    antoine = lambda A, B, C, t: (10 ** (A - B / (C + t))) * 133.322 / 1000  # kPa
    data = {
        "Water":    dict(den=1000 - ((T - 4)**2)/200 + np.random.normal(0, .5, n),
                         vis=1.002*np.exp(-.025*(T-20)) + np.random.normal(0, .005, n),
                         st =75.6 - .14*T + np.random.normal(0, .1, n),
                         vp =antoine(8.07131, 1730.63, 233.426, T) + np.random.normal(0, .1, n)),
        "Ethanol":  dict(den=806 - .85*T + np.random.normal(0, .5, n),
                         vis=1.2*np.exp(-.03*(T-20)) + np.random.normal(0, .01, n),
                         st =24.0 - .08*T + np.random.normal(0, .1, n),
                         vp =antoine(8.20417, 1642.89, 230.300, T) + np.random.normal(0, .5, n)),
        "Acetone":  dict(den=810 - 1.0*T + np.random.normal(0, .5, n),
                         vis=.32*np.exp(-.015*(T-20)) + np.random.normal(0, .002, n),
                         st =26.0 - .1*T + np.random.normal(0, .1, n),
                         vp =antoine(7.02447, 1161.0, 224.0, T) + np.random.normal(0, 1.0, n)),
        "Glycerin": dict(den=1275 - .6*T + np.random.normal(0, .5, n),
                         vis=1412*np.exp(-.08*(T-20)) + np.random.normal(0, 5.0, n),
                         st =64.0 - .06*T + np.random.normal(0, .2, n),
                         vp =.001*np.exp(.05*T) + np.random.normal(0, .0001, n)),
    }
    rows = {k: [] for k in ["Fluid_Type", "Temperature_C", *PROPS]}
    for fluid, d in data.items():
        rows["Fluid_Type"]           += [fluid]*n
        rows["Temperature_C"]        += list(T)
        rows["Density_kg_m3"]        += list(d["den"])
        rows["Viscosity_mPa_s"]      += list(d["vis"])
        rows["Surface_Tension_mN_m"] += list(d["st"])
        rows["Vapor_Pressure_kPa"]   += list(d["vp"])
    df = pd.DataFrame(rows)
    df[PROPS] = df[PROPS].clip(lower=0.0001)
    return df

@st.cache_resource
def train_all(df):
    models, metrics = {}, {}
    for fluid in df["Fluid_Type"].unique():
        models[fluid], metrics[fluid] = {}, {}
        sub = df[df["Fluid_Type"] == fluid]
        X = sub[["Temperature_C"]].values
        for prop in PROPS:
            y = sub[prop].values
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
            poly = make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(X, y)
            models[fluid][prop] = {"RandomForest": rf, "Polynomial3": poly}
            metrics[fluid][prop] = {
                "RandomForest": {"R2": r2_score(y, rf.predict(X)),
                                 "RMSE": np.sqrt(mean_squared_error(y, rf.predict(X)))},
                "Polynomial3":  {"R2": r2_score(y, poly.predict(X)),
                                 "RMSE": np.sqrt(mean_squared_error(y, poly.predict(X)))},
            }
    return models, metrics

# ─────────────────────────────────────────────────────────────────────────────
# 2. 외삽 가드 (보고서 결론에서 요구한 예외처리 — 원본 코드에는 미구현)
# ─────────────────────────────────────────────────────────────────────────────
def predict_safe(models, fluid, prop, model_name, T_c):
    """학습범위 [0,100]°C 이탈 시 (값, 경고문) 반환."""
    warn = None
    if T_c < TRAIN_LO or T_c > TRAIN_HI:
        tail = (" Random Forest 는 경계 학습값으로 평탄화되어 추세를 연장하지 못합니다."
                if model_name == "RandomForest" else
                " 3차 다항회귀는 경계 밖에서 발산(Runge 유사 왜곡)할 수 있습니다.")
        warn = (f"입력 온도 {T_c:.1f}°C 가 학습 범위 [{TRAIN_LO:.0f}, {TRAIN_HI:.0f}]°C 를 "
                f"벗어났습니다. 외삽 결과는 물리적 신뢰성이 보장되지 않습니다." + tail)
    val = float(models[fluid][prop][model_name].predict([[T_c]])[0])
    return val, warn

# ─────────────────────────────────────────────────────────────────────────────
# 3. 앱 본문
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI 유체물성 예측 (Ch1-1)", layout="wide")
st.markdown("""
<style>
div[data-testid="stMetricLabel"]{white-space:normal!important;overflow:visible!important;}
div[data-testid="stMetricValue"]{font-size:1.4rem!important;}
</style>
""", unsafe_allow_html=True)

df = build_dataset()
models, metrics = train_all(df)

st.sidebar.title("AI 융합 유체역학 · Ch1-1")
menu = st.sidebar.radio("프로젝트 단계", [
    "Step 0: 데이터셋 및 출처",
    "Step 1: AI-Physics 물성 예측",
    "Step 2: 오차 분석 & 외삽 한계",
])

# ── Step 0 ───────────────────────────────────────────────────────────────────
if menu == "Step 0: 데이터셋 및 출처":
    st.title("학습 데이터셋 및 구축 출처")
    st.info(
        "**물성 데이터** : NIST Chemistry WebBook 경험식(Antoine 식, Andrade 식 등) 기반 "
        "4종 유체(Water·Ethanol·Acetone·Glycerin) × 200점 = **800행**.\n\n"
        "온도(0–100°C)에 대한 밀도·점성계수·표면장력·증기압을 물리식으로 생성하고 "
        "실험 오차를 모사한 정규분포 노이즈를 부가하였다."
    )
    st.write(f"**총 데이터 개수 : {len(df)} 행 / {df['Fluid_Type'].nunique()} 유체**")
    st.dataframe(df, use_container_width=True, height=420)
    st.caption("증기압 생성식 P = 10^(A − B/(C+T)) × 133.322/1000 [kPa] (Antoine), "
               "점성 μ = μ₀·exp(−k(T−T₀)) (Andrade 형).")

# ── Step 1 ───────────────────────────────────────────────────────────────────
elif menu == "Step 1: AI-Physics 물성 예측":
    st.title("AI-Physics 기반 유체 물성 예측")
    col1, col2 = st.columns([4, 6])

    with col1:
        fluid = st.selectbox("대상 유체", df["Fluid_Type"].unique())
        prop  = st.selectbox("예측 물성치", PROPS, format_func=lambda p: f"{NAME_KO[p]} ({p})")
        model_name = st.radio("AI 모델", ["RandomForest", "Polynomial3"],
                              format_func=lambda m: MODELS_KO[m])

        # ── 단위 선택 (인터페이스 루브릭 A 요건) ──
        t_unit = st.selectbox("온도 단위", ["°C", "°F", "K"])
        p_unit = st.selectbox(f"{NAME_KO[prop]} 단위", list(PROP_UNITS[prop].keys()))
        scale, offset = PROP_UNITS[prop][p_unit]

        lo_disp, hi_disp = C_to(t_unit, 0.0), C_to(t_unit, 100.0)
        # 학습범위 밖 입력도 허용해 외삽 가드를 시연
        in_lo = float(min(lo_disp, hi_disp) - abs(hi_disp - lo_disp) * 0.4)
        in_hi = float(max(lo_disp, hi_disp) + abs(hi_disp - lo_disp) * 0.4)
        T_disp = st.slider(f"작동 온도 ({t_unit})", in_lo, in_hi,
                           float(C_to(t_unit, 25.0)))
        T_c = to_C(t_unit, T_disp)

    with col2:
        val_c, warn = predict_safe(models, fluid, prop, model_name, T_c)
        pred = val_c * scale + offset

        sub = df[df["Fluid_Type"] == fluid]
        idx = (sub["Temperature_C"] - T_c).abs().idxmin()
        actual = sub.loc[idx, prop] * scale + offset
        err = pred - actual
        err_pct = (err / actual * 100) if actual != 0 else 0.0

        st.subheader(f"{fluid} · {NAME_KO[prop]} 예측 결과")
        if warn:
            st.warning("⚠️ " + warn)

        a, b, c = st.columns(3)
        a.metric(f"예측값 ({MODELS_KO[model_name]})", f"{pred:.4f} {p_unit}")
        b.metric("실측값 (NIST 최근접)", f"{actual:.4f} {p_unit}")
        c.metric("오차", f"{err:+.4f}", delta=f"{err_pct:+.2f}%", delta_color="inverse")

        m = metrics[fluid][prop][model_name]
        d, e = st.columns(2)
        d.metric("결정계수 R²", f"{m['R2']:.4f}")
        e.metric(f"RMSE ({prop} 기준단위)", f"{m['RMSE']:.4f}")

        fig, ax = plt.subplots(figsize=(9, 4.6))
        Tc_curve = np.linspace(0, 100, 300).reshape(-1, 1)
        curve = models[fluid][prop][model_name].predict(Tc_curve) * scale + offset
        Tc_disp = C_to(t_unit, Tc_curve.ravel())
        ax.scatter(C_to(t_unit, sub["Temperature_C"]), sub[prop]*scale+offset,
                   s=10, c="grey", alpha=.35, label="NIST data")
        ax.plot(Tc_disp, curve, c="#c0392b", lw=2, label=f"AI ({MODELS_KO[model_name]})")
        ax.axvline(T_disp, c="#2471a3", ls="--", lw=1.2)
        ax.set_xlabel(f"Temperature ({t_unit})"); ax.set_ylabel(f"{NAME_KO[prop]} ({p_unit})")
        ax.set_title(f"{fluid} — Temperature vs {NAME_KO[prop]}")
        ax.legend(fontsize=8)
        st.pyplot(fig)

# ── Step 2 ───────────────────────────────────────────────────────────────────
elif menu == "Step 2: 오차 분석 & 외삽 한계":
    st.title("오차 분석 및 외삽(Extrapolation) 한계")
    st.markdown(
        "- **보간 영역(0–100°C)** : RandomForest 는 노이즈까지 국소 적합하여 잔차가 작고, "
        "3차 다항회귀는 물리 추세선을 따른다.\n"
        "- **외삽 영역(>100°C)** : RandomForest 는 경계값으로 평탄화되어 추세를 연장하지 못하고, "
        "다항회귀는 발산(Runge 유사 왜곡)한다 → 실무 적용 시 외삽 가드가 필수."
    )

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    f, p = "Water", "Density_kg_m3"; sub = df[df.Fluid_Type == f]
    Tc = np.linspace(0, 100, 300).reshape(-1, 1)
    ax[0,0].scatter(sub.Temperature_C, sub[p], s=8, c="grey", alpha=.4, label="NIST")
    ax[0,0].plot(Tc, models[f][p]["RandomForest"].predict(Tc), c="#c0392b", lw=2, label="RF")
    ax[0,0].plot(Tc, models[f][p]["Polynomial3"].predict(Tc), c="#2471a3", lw=2, ls="--", label="Poly(3)")
    ax[0,0].set_title(f"(a) {f} 밀도 — 완만 거동"); ax[0,0].legend(fontsize=8)

    f, p = "Glycerin", "Viscosity_mPa_s"; sub = df[df.Fluid_Type == f]
    ax[0,1].scatter(sub.Temperature_C, sub[p], s=8, c="grey", alpha=.4, label="NIST")
    ax[0,1].plot(Tc, models[f][p]["RandomForest"].predict(Tc), c="#c0392b", lw=2, label="RF")
    ax[0,1].plot(Tc, models[f][p]["Polynomial3"].predict(Tc), c="#2471a3", lw=2, ls="--", label="Poly(3)")
    ax[0,1].set_title(f"(b) {f} 점성 — 강한 지수 거동"); ax[0,1].legend(fontsize=8)

    f, p = "Water", "Density_kg_m3"; sub = df[df.Fluid_Type == f]
    Xtr = sub[["Temperature_C"]].values; y = sub[p].values
    ax[1,0].scatter(sub.Temperature_C, y - models[f][p]["RandomForest"].predict(Xtr),
                    s=8, c="#c0392b", alpha=.5, label="RF 잔차")
    ax[1,0].scatter(sub.Temperature_C, y - models[f][p]["Polynomial3"].predict(Xtr),
                    s=8, c="#2471a3", alpha=.5, label="Poly 잔차")
    ax[1,0].axhline(0, c="k", lw=.8); ax[1,0].set_title("(c) 잔차 분포"); ax[1,0].legend(fontsize=8)

    f, p = "Acetone", "Vapor_Pressure_kPa"; sub = df[df.Fluid_Type == f]
    Tex = np.linspace(0, 140, 400).reshape(-1, 1)
    ax[1,1].scatter(sub.Temperature_C, sub[p], s=8, c="grey", alpha=.4, label="NIST(0-100)")
    ax[1,1].plot(Tex, models[f][p]["RandomForest"].predict(Tex), c="#c0392b", lw=2, label="RF(평탄화)")
    ax[1,1].plot(Tex, models[f][p]["Polynomial3"].predict(Tex), c="#2471a3", lw=2, ls="--", label="Poly(발산)")
    ax[1,1].axvspan(100, 140, color="orange", alpha=.12); ax[1,1].axvline(100, c="orange", lw=1.5)
    ax[1,1].set_title("(d) 외삽 한계 — 가드 작동 구간"); ax[1,1].legend(fontsize=8)

    fig.tight_layout(); st.pyplot(fig)
