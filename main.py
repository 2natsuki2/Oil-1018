import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import japanize_matplotlib # グラフの日本語化

# -----------------------------------------------------------
# 1. ファイル任意選択のためのGUI
# -----------------------------------------------------------
# ファイル選択ウィンドウの準備
root = tk.Tk()
root.withdraw() # 小さなウィンドウを隠す

# パスを取得
filepath = filedialog.askopenfilename(
    title="分析したいCSVファイルを選択してください",
    filetypes=[("CSVファイル", "*.csv")] 
)

# -----------------------------------------------------------
# 2. ファイルが選ばれた場合のみ、AIの処理を開始
# -----------------------------------------------------------
if filepath:
    print(f"選択されたファイル: {filepath}")

    # --- データの読み込みと前処理 ---
    print("\n--- データの読み込みと前処理中... ---")
    df = pd.read_csv(filepath)
    # 'date'列を日付として扱えるように変換し、行のラベル（インデックス）に設定
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # --- 改善後モデル（工夫あり）の準備 ---
    print("\n--- 改善後モデルを学習中... ---")

    # 工夫①：カレンダーと時計のヒントを追加
    df_improved = df.copy()
    df_improved['month'] = df_improved.index.month
    df_improved['hour'] = df_improved.index.hour

    # 工夫②：1時間前の温度のヒントを追加
    df_improved['OT_lag_1'] = df_improved['OT'].shift(1) 
    df_improved = df_improved.dropna() 

    # 「答え（OT）」と「ヒント（それ以外）」を分ける
    target = 'OT'
    features_imp = [col for col in df_improved.columns if col != 'OT']
    
    # データを勉強用とテスト用に分ける
    train_val_df_imp, test_df_imp = train_test_split(df_improved, test_size=0.2, shuffle=False)
    train_df_imp, val_df_imp = train_test_split(train_val_df_imp, test_size=0.125, shuffle=False)

    X_train_imp, y_train_imp = train_df_imp[features_imp], train_df_imp[target]
    X_val_imp, y_val_imp = val_df_imp[features_imp], val_df_imp[target]
    X_test_imp, y_test_imp = test_df_imp[features_imp], test_df_imp[target]

    # 数値の大きさを揃える（スケーリング）
    scaler_imp = StandardScaler()
    X_train_scaled_imp = scaler_imp.fit_transform(X_train_imp)
    X_val_scaled_imp = scaler_imp.transform(X_val_imp)
    X_test_scaled_imp = scaler_imp.transform(X_test_imp)

    # 改善後モデルの学習
    lgb_model_imp = lgb.LGBMRegressor(random_state=42)
    lgb_model_imp.fit(X_train_scaled_imp, y_train_imp,
                      eval_set=[(X_val_scaled_imp, y_val_imp)],
                      eval_metric='mae',
                      callbacks=[lgb.early_stopping(10, verbose=False)])

    # 改善後モデルの答え合わせ
    y_pred_lgb_imp = lgb_model_imp.predict(X_test_scaled_imp)
    mae_lgb_imp = mean_absolute_error(y_test_imp, y_pred_lgb_imp)
    print(f"最終モデルの平均誤差: {mae_lgb_imp:.3f}℃")

    # --- グラフの表示 ---
    print("\n--- 最終結果のグラフを生成中... ---")

    # グラフ: 改善後の結果（青と赤）
    plt.figure(figsize=(15, 7))
    plt.plot(y_test_imp.index, y_test_imp, label='実績値 (答え)', color='blue', alpha=0.8)
    plt.plot(y_test_imp.index, y_pred_lgb_imp, label=f'AIの予測 (MAE:{mae_lgb_imp:.3f})', color='red', linestyle='--')
    plt.title('AI予測結果')
    plt.xlabel('日付')
    plt.ylabel('オイル温度 (OT)')
    plt.legend()
    plt.grid(True)
    plt.show() # グラフを表示
    
    print("\n--- 全ての処理が完了しました ---")

else:
    # ファイルが選択されなかった場合
    print("ファイルが選択されなかったので、処理を中断しました。")
