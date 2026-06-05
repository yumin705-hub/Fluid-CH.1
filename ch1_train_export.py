"""
==============================================================================
 프로젝트 1-1 — AI 기반 유체물성 예측 : 모델 학습·내보내기 (헤드리스)
 -----------------------------------------------------------------------------
 보고서 본문의 데이터 파이프라인을 그대로 재현하여
   (1) NIST 경험식 기반 4 유체 × 200점 = 800행 데이터셋 생성
   (2) 물성 4종 × 모델 2종(RandomForest, 3차 다항회귀) 학습
   (3) R²·RMSE 산출  (4) 학습 모델을 ch1_models.pkl 로 저장(joblib)
   (5) 오차분석 + 외삽 한계 시각화를 ch1_error_analysis.png 로 저장
 실행 :  python ch1_train_export.py
==============================================================================
"""
import numpy as np, pandas as pd, os, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

OUT = os.path.dirname(os.path.abspath(__file__))
PROPS = ["Density_kg_m3", "Viscosity_mPa_s", "Surface_Tension_mN_m", "Vapor_Pressure_kPa"]
UNIT  = {"Density_kg_m3": "kg/m^3", "Viscosity_mPa_s": "mPa.s",
         "Surface_Tension_mN_m": "mN/m", "Vapor_Pressure_kPa": "kPa"}
TRAIN_LO, TRAIN_HI = 0.0, 100.0   # 학습 온도 범위 (외삽 가드 기준)

# ─────────────────────────────────────────────────────────────────────────────
# 1. NIST 경험식 기반 데이터 생성 (보고서 2절과 동일)
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(path):
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
    rows = {"Fluid_Type": [], "Temperature_C": [],
            "Density_kg_m3": [], "Viscosity_mPa_s": [],
            "Surface_Tension_mN_m": [], "Vapor_Pressure_kPa": []}
    for fluid, d in data.items():
        rows["Fluid_Type"]          += [fluid]*n
        rows["Temperature_C"]       += list(T)
        rows["Density_kg_m3"]       += list(d["den"])
        rows["Viscosity_mPa_s"]     += list(d["vis"])
        rows["Surface_Tension_mN_m"]+= list(d["st"])
        rows["Vapor_Pressure_kPa"]  += list(d["vp"])
    df = pd.DataFrame(rows)
    df[PROPS] = df[PROPS].clip(lower=0.0001)
    df.to_csv(path, index=False)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. 모델 학습 (RandomForest + 3차 다항회귀)
# ─────────────────────────────────────────────────────────────────────────────
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
# 3. 외삽 가드 (보고서 결론에서 요구한 예외처리 — 원본 코드에는 미구현)
# ─────────────────────────────────────────────────────────────────────────────
def predict_safe(models, fluid, prop, model_name, T):
    """학습 범위[0,100]°C 를 벗어나면 경고와 함께 외삽 결과를 표시."""
    warn = None
    if T < TRAIN_LO or T > TRAIN_HI:
        warn = (f"[경고] 입력 온도 {T:.1f}°C 가 학습 범위 "
                f"[{TRAIN_LO:.0f}, {TRAIN_HI:.0f}]°C 를 벗어났습니다. "
                f"외삽 결과는 물리적 신뢰성이 보장되지 않습니다"
                + (" (Random Forest 는 경계값으로 평탄화됨)."
                   if model_name == "RandomForest" else
                   " (다항회귀는 발산 가능)."))
    val = models[fluid][prop][model_name].predict([[T]])[0]
    return val, warn

# ─────────────────────────────────────────────────────────────────────────────
# 4. 오차분석 + 외삽 한계 시각화
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(df, models, metrics, path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) 완만 거동: Water 밀도 — RF vs Poly 적합
    f, p = "Water", "Density_kg_m3"
    sub = df[df["Fluid_Type"] == f]; T = np.linspace(0, 100, 300).reshape(-1, 1)
    ax[0,0].scatter(sub["Temperature_C"], sub[p], s=8, c="grey", alpha=.4, label="NIST data")
    ax[0,0].plot(T, models[f][p]["RandomForest"].predict(T), c="#c0392b", lw=2, label="Random Forest")
    ax[0,0].plot(T, models[f][p]["Polynomial3"].predict(T), c="#2471a3", lw=2, ls="--", label="Polynomial(3)")
    ax[0,0].set_title(f"(a) {f} Density — smooth regime\nRF R2={metrics[f][p]['RandomForest']['R2']:.4f}, "
                      f"Poly R2={metrics[f][p]['Polynomial3']['R2']:.4f}")
    ax[0,0].set_xlabel("Temperature (C)"); ax[0,0].set_ylabel(UNIT[p]); ax[0,0].legend(fontsize=8)

    # (b) 강비선형: Glycerin 점성 — 지수 거동
    f, p = "Glycerin", "Viscosity_mPa_s"
    sub = df[df["Fluid_Type"] == f]
    ax[0,1].scatter(sub["Temperature_C"], sub[p], s=8, c="grey", alpha=.4, label="NIST data")
    ax[0,1].plot(T, models[f][p]["RandomForest"].predict(T), c="#c0392b", lw=2, label="Random Forest")
    ax[0,1].plot(T, models[f][p]["Polynomial3"].predict(T), c="#2471a3", lw=2, ls="--", label="Polynomial(3)")
    ax[0,1].set_title(f"(b) {f} Viscosity — strong exponential\nRF R2={metrics[f][p]['RandomForest']['R2']:.4f}, "
                      f"Poly R2={metrics[f][p]['Polynomial3']['R2']:.4f}")
    ax[0,1].set_xlabel("Temperature (C)"); ax[0,1].set_ylabel(UNIT[p]); ax[0,1].legend(fontsize=8)

    # (c) 잔차 분포: Water 밀도
    f, p = "Water", "Density_kg_m3"; sub = df[df["Fluid_Type"] == f]
    Xtr = sub[["Temperature_C"]].values; y = sub[p].values
    ax[1,0].scatter(sub["Temperature_C"], y - models[f][p]["RandomForest"].predict(Xtr),
                    s=8, c="#c0392b", alpha=.5, label="RF residual")
    ax[1,0].scatter(sub["Temperature_C"], y - models[f][p]["Polynomial3"].predict(Xtr),
                    s=8, c="#2471a3", alpha=.5, label="Poly residual")
    ax[1,0].axhline(0, c="k", lw=.8)
    ax[1,0].set_title("(c) Residuals — Poly follows physical trend,\nRF fits noise locally")
    ax[1,0].set_xlabel("Temperature (C)"); ax[1,0].set_ylabel("residual"); ax[1,0].legend(fontsize=8)

    # (d) 외삽 한계: Acetone 증기압을 0~140°C 로 확장 (학습 0~100)
    f, p = "Acetone", "Vapor_Pressure_kPa"; sub = df[df["Fluid_Type"] == f]
    Tex = np.linspace(0, 140, 400).reshape(-1, 1)
    ax[1,1].scatter(sub["Temperature_C"], sub[p], s=8, c="grey", alpha=.4, label="NIST data (0-100)")
    ax[1,1].plot(Tex, models[f][p]["RandomForest"].predict(Tex), c="#c0392b", lw=2,
                 label="RF (flatlines >100)")
    ax[1,1].plot(Tex, models[f][p]["Polynomial3"].predict(Tex), c="#2471a3", lw=2, ls="--",
                 label="Poly (may diverge)")
    ax[1,1].axvspan(100, 140, color="orange", alpha=.12)
    ax[1,1].axvline(100, c="orange", lw=1.5)
    ax[1,1].text(101, ax[1,1].get_ylim()[1]*0.5, "extrapolation\n(guard triggers)",
                 fontsize=8, color="darkorange")
    ax[1,1].set_title("(d) Extrapolation failure beyond training range")
    ax[1,1].set_xlabel("Temperature (C)"); ax[1,1].set_ylabel(UNIT[p]); ax[1,1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print(" 프로젝트 1-1 모델 학습·내보내기")
    print("=" * 72)
    csv_path = os.path.join(OUT, "fluid_properties.csv")
    df = build_dataset(csv_path)
    print(f"[1] 데이터셋 {len(df)} 행 생성 → fluid_properties.csv "
          f"({df['Fluid_Type'].nunique()} 유체 × 200점)")

    models, metrics = train_all(df)
    print(f"\n[2] 모델 성능 (R² / RMSE)")
    print(f"    {'유체':<10}{'물성':<22}{'RF R²':>9}{'Poly R²':>10}")
    rf_r2 = []
    for f in metrics:
        for p in PROPS:
            r1 = metrics[f][p]["RandomForest"]["R2"]; r2 = metrics[f][p]["Polynomial3"]["R2"]
            rf_r2.append(r1)
            print(f"    {f:<10}{p:<22}{r1:>9.4f}{r2:>10.4f}")
    print(f"    → Random Forest 평균 R² = {np.mean(rf_r2):.4f}  (루브릭 기준 R² > 0.98)")

    joblib.dump({"models": models, "metrics": metrics,
                 "train_range": (TRAIN_LO, TRAIN_HI)},
                os.path.join(OUT, "ch1_models.pkl"))
    make_figure(df, models, metrics, os.path.join(OUT, "ch1_error_analysis.png"))

    # 외삽 가드 시연
    print(f"\n[3] 외삽 가드 시연 (Acetone 증기압)")
    for T in (25.0, 110.0):
        v, w = predict_safe(models, "Acetone", "Vapor_Pressure_kPa", "RandomForest", T)
        print(f"    T={T:>6.1f}°C → 예측 {v:.3f} kPa" + (f"\n        {w}" if w else "  [정상 범위]"))

    print(f"\n[저장 완료] ch1_models.pkl · ch1_error_analysis.png · fluid_properties.csv")
    print("=" * 72)
