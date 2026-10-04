import sqlite3
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

class MusicClusteringModel:
    def __init__(self, db_path='music.db', model_folder='models', n_clusters=5):
        self.db_path = db_path
        self.model_folder = Path(model_folder)
        self.n_clusters = n_clusters
        self.pipeline = None
        self.kmeans = None
        self.df = None
        self.original_ids = None # 用來記錄過濾後的 ID，以便寫回資料庫
        
        # 確保模型資料夾存在
        self.model_folder.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """
        1. 連接資料庫
        2. 讀取資料
        3. 過濾異常資料 (BPM <= 0)
        """
        print("🚀 [Step 1] Loading data from database...")
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT id, title, artist, bpm, energy, danceability, happiness, 
            acousticness, instrumentalness, liveness, speechiness, loudness,
            key, camelot
        FROM songs
        """
        raw_df = pd.read_sql(query, conn)
        conn.close()

        # 簡單統計原始數量
        initial_count = len(raw_df)
        
        # --- 資料清洗 ---
        # 1. 過濾掉 BPM 為 0 或負數的資料
        self.df = raw_df[raw_df['bpm'] > 0].copy()
        
        # 2. 填補數值欄位的 NaN 為 0 (以防萬一)
        num_cols = ['energy', 'danceability', 'happiness', 'acousticness', 
                   'instrumentalness', 'liveness', 'speechiness', 'loudness']
        self.df[num_cols] = self.df[num_cols].fillna(0)
        
        # 3. 填補類別欄位的缺失值為 "Unknown"
        cat_cols = ['key', 'camelot']
        self.df[cat_cols] = self.df[cat_cols].fillna('Unknown')

        dropped_count = initial_count - len(self.df)
        print(f"   Original: {initial_count}, Dropped (BPM<=0): {dropped_count}, Remaining: {len(self.df)}")

        # 保存 id 列表，因為之後 df 會被轉換成 numpy array，我們需要保留 id 來寫回資料庫
        self.original_ids = self.df['id'].values

    def build_pipeline(self):
        """
        建立資料處理與模型訓練的 Pipeline
        這樣可以確保「訓練時」跟「預測新資料時」用的是同一套標準
        """
        print("⚙️ [Step 2] Building preprocessing pipeline...")
        
        # 定義數值特徵
        numeric_features = ['bpm', 'energy', 'danceability', 'happiness', 
                            'acousticness', 'instrumentalness', 'liveness', 
                            'speechiness', 'loudness']
        
        # 定義類別特徵 (要轉 One-Hot 的)
        categorical_features = ['key', 'camelot']

        # 建立轉換器
        preprocessor = ColumnTransformer(
            transformers=[
                # 數值欄位 -> 標準化 (StandardScaler)
                ('num', StandardScaler(), numeric_features),
                
                # 類別欄位 -> 獨熱編碼 (OneHotEncoder)
                # handle_unknown='ignore': 遇到沒看過的 Key 不會報錯，全填 0
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # 建立完整的 Pipeline (前處理 + 模型)
        # 雖然 KMeans 也可以單獨跑，但包在一起管理比較方便
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('kmeans', KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10))
        ])

    def train(self):
        """
        訓練模型
        """
        print(f"🧠 [Step 3] Training K-Means model (k={self.n_clusters})...")
        if self.df is None:
            raise ValueError("Data not loaded! Call load_data() first.")

        # 訓練並進行預測
        # fit_predict 會跑完整個流程：標準化 -> OneHot -> KMeans
        self.df['cluster'] = self.pipeline.fit_predict(self.df)
        
        print("   Training complete.")

    def save_model(self):
        """
        儲存 Pipeline (包含 Scaler, Encoder 和 KMeans 模型)
        """
        print("💾 [Step 4] Saving model...")
        model_file = self.model_folder / "music_clustering_pipeline.pkl"
        joblib.dump(self.pipeline, model_file)
        print(f"   Model saved to {model_file}")

    def update_database(self):
        """
        將分群結果寫回 SQLite
        """
        print("📝 [Step 5] Updating database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 準備更新資料: (cluster_id, song_id)
        # 注意：我們只更新那些有參與訓練的資料 (BPM > 0)
        update_data = list(zip(
            self.df['cluster'].astype(int).tolist(), 
            self.original_ids.tolist()
        ))

        # 批量更新
        cursor.executemany("UPDATE songs SET cluster = ? WHERE id = ?", update_data)
        conn.commit()
        conn.close()
        print(f"   Updated {len(update_data)} records in database.")

    def show_statistics(self):
        """
        顯示統計結果與範例歌曲
        """
        print("\n" + "="*40)
        print("📊 Clustering Statistics")
        print("="*40)
        print(self.df['cluster'].value_counts().sort_index())

        print("\n🎵 Representative Songs per Cluster:")
        for i in range(self.n_clusters):
            print(f"\n[Cluster {i}]")
            # 隨機抽樣 3 首歌顯示
            sample = self.df[self.df['cluster'] == i].head(3)
            for _, row in sample.iterrows():
                print(f"  - {row['title']} ({row['artist']}) | BPM: {row['bpm']} | Key: {row['key']}")

if __name__ == "__main__":
    # 使用方式
    model = MusicClusteringModel(n_clusters=16)
    model.load_data()
    model.build_pipeline()
    model.train()
    model.save_model()
    model.update_database()
    model.show_statistics()