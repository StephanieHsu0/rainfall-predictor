import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型定義
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class SimpleTimesNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleTimesNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear2(self.relu(self.linear1(x)))

# Streamlit 介面
st.title("🌧️ Hourly Rainfall Prediction Dashboard")
st.sidebar.title("模型選擇與設定")

model_type = st.sidebar.selectbox("請選擇模型：", ["LSTM", "TimesNet"])
show_samples = st.sidebar.slider("顯示前幾筆預測結果", 50, 500, 200, step=50)

# 載入資料
try:
    X = np.load("data/processed/X_test.npy")
    Y = np.load("data/processed/Y_test.npy")
except FileNotFoundError:
    st.error("❌ 找不到測試資料，請確認路徑為 data/processed/")
    st.stop()

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

input_dim = X.shape[2]
seq_len = X.shape[1]
output_dim = Y.shape[1]

# 載入模型
try:
    if model_type == "LSTM":
        model = LSTMModel(input_dim, 64, output_dim).to(device)
        model.load_state_dict(torch.load("models/lstm_model.pth", map_location=device))
    else:
        model = SimpleTimesNet(seq_len * input_dim, 128, output_dim).to(device)
        model.load_state_dict(torch.load("models/timesnet_model.pth", map_location=device))
except FileNotFoundError:
    st.error("❌ 找不到模型權重，請確認路徑為 models/lstm_model.pth 或 models/timesnet_model.pth")
    st.stop()

# 預測
model.eval()
with torch.no_grad():
    preds = model(X_tensor).cpu().numpy()
    true = Y_tensor.cpu().numpy()

# 評估
actual = true[:, 0]
predicted = preds[:, 0]
mask = ~np.isnan(actual) & ~np.isnan(predicted)
actual = actual[mask]
predicted = predicted[mask]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

st.subheader("📈 模型效能評估")
st.write(f"**MAE**: {mae:.4f} mm")
st.write(f"**RMSE**: {rmse:.4f} mm")
st.write(f"**R² Score**: {r2:.4f}")

# 圖表
st.subheader("📊 預測圖 vs 實際降雨")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(actual[:show_samples], label='Actual')
ax.plot(predicted[:show_samples], label='Predicted', linestyle='--')
ax.set_title(f"{model_type} Prediction (First {show_samples} Samples)")
ax.set_xlabel("Time Step")
ax.set_ylabel("Rainfall (mm)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# 表格
st.subheader("🔍 前幾筆預測結果")
df_show = pd.DataFrame({
    "Actual": actual[:show_samples],
    "Predicted": predicted[:show_samples],
    "Error": actual[:show_samples] - predicted[:show_samples]
})
st.dataframe(df_show.style.format("{:.3f}"))
