import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

if platform.system() == "Darwin":  #
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="NanumGothic")

fe = fm.FontEntry(
    fname=r"/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # ttf 파일이 저장되어 있는 경로
    name="NanumGothic",
)  # 원하는 폰트 설정
fm.fontManager.ttflist.insert(0, fe)  # Matplotlib에 폰트 추가

plt.rcParams.update({"font.size": 18, "font.family": "NanumGothic"})  # 폰트 설정

plt.rcParams["axes.unicode_minus"] = False

st.markdown("## XGBoost Example")
df = pd.read_csv("input.csv")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### Select Target Variable")
col = st.selectbox("Target 변수를 선택하세요", df.columns[1:])
st.dataframe(df[col])

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### Select Input Variables")
cols = st.multiselect("복수의 컬럼을 선택하세요", df.columns[1:])
# filtered_iris = iris.loc[:, cols]
st.dataframe(df[cols])

Xt, Xts, yt, yts = train_test_split(df[cols], df[col], test_size=0.2, shuffle=False)

max_depth = st.slider("max_depth:", 0, 20, value=3)
n_estimators = st.slider("n_estimators:", 0, 500, value=50)
learning_rate = st.slider("learning_rate:", 0.0, 1.0, value=0.1)
subsample = st.slider("subsample:", 0.0, 1.0, value=0.8)

xgb = XGBRegressor(
    max_depth=max_depth,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    subsample=subsample,
    random_state=2,
    n_jobs=-1,
)

xgb.fit(Xt, yt)

yt_pred = xgb.predict(Xt)
yts_pred = xgb.predict(Xts)

mse_train = mean_squared_error(10**yt, 10**yt_pred)
mse_test = mean_squared_error(10**yts, 10**yts_pred)
st.write(f"학습 데이터 MSE: {mse_train}")
st.write(f"테스트 데이터 MSE: {mse_test}")

r2_train = r2_score(10**yt, 10**yt_pred)
r2_test = r2_score(10**yts, 10**yts_pred)
st.write(f"학습 데이터 R2: {r2_train}")
st.write(f"테스트 데이터 R2: {r2_test}")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(Xt["로그 원수 탁도"], yt, s=3, label="학습 데이터 (실제)")
ax.scatter(Xt["로그 원수 탁도"], yt_pred, s=3, label="학습 데이터 (예측)", c="r")
ax.grid()
ax.legend(fontsize=13)
ax.set_xlabel("로그 원수 탁도")
ax.set_ylabel("로그 응집제 주입률")
ax.set_title(
    rf"학습 데이터  MSE: {round(mse_train, 4)}, $R^2$: {round(r2_train, 2)}",
    fontsize=18,
)

ax = axes[1]
ax.scatter(Xts["로그 원수 탁도"], yts, s=3, label="테스트 데이터 (실제)")
ax.scatter(Xts["로그 원수 탁도"], yts_pred, s=3, label="테스트 데이터 (예측)", c="r")
ax.grid()
ax.legend(fontsize=13)
ax.set_xlabel("로그 원수 탁도")
ax.set_ylabel("로그 응집제 주입률")
ax.set_title(
    rf"테스트 데이터  MSE: {round(mse_test, 4)}, $R^2$: {round(r2_test, 2)}",
    fontsize=18,
)

st.pyplot(fig)
