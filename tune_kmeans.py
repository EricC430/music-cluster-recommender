import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DB_PATH = 'music.db'

def load_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT bpm, energy, danceability, happiness, 
           acousticness, instrumentalness, liveness, speechiness, loudness
    FROM songs
    """
    df = pd.read_sql(query, conn)
    conn.close()
    # 過濾掉壞資料
    df = df[df['bpm'] > 0].fillna(0)
    return df

def analyze_k(df):
    """
    比較 StandardScaler vs PowerTransformer
    並畫出 Elbow 和 Silhouette 圖表
    """
    features = df.columns.tolist()
    
    # 設定兩種預處理方式來比較
    scalers = {
        "StandardScaler (原本)": StandardScaler(),
        "PowerTransformer (推薦)": PowerTransformer(method='yeo-johnson') 
    }
    
    range_n_clusters = range(2, 21) # 測試 K=2 到 K=10

    plt.figure(figsize=(15, 6))

    for i, (name, scaler) in enumerate(scalers.items()):
        print(f"\n🧪 Testing with {name}...")
        
        # 資料轉換
        X_scaled = scaler.fit_transform(df)
        
        inertias = []
        silhouette_scores = []
        
        for k in range_n_clusters:
            # n_init=10 確保穩定性
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            
            inertias.append(kmeans.inertia_)
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
            
            print(f"   K={k}: Silhouette={score:.4f}")

        # 繪圖 - Elbow Method
        plt.subplot(2, 2, i*2 + 1)
        plt.plot(range_n_clusters, inertias, 'bo-')
        plt.title(f'{name} - Elbow Method (Inertia)')
        plt.xlabel('K (Number of clusters)')
        plt.ylabel('Inertia (Lower is better)')
        
        # 繪圖 - Silhouette Score
        plt.subplot(2, 2, i*2 + 2)
        plt.plot(range_n_clusters, silhouette_scores, 'rs-')
        plt.title(f'{name} - Silhouette Score')
        plt.xlabel('K (Number of clusters)')
        plt.ylabel('Score (Higher is better)')

    plt.tight_layout()
    plt.show()
    print("\n✅ Analysis Complete. Check the plots!")

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        analyze_k(df)
    else:
        print("❌ 資料庫是空的")