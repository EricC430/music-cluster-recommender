import sqlite3
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.utils
import json
import plotly.graph_objects as go
import re

# 設定檔案路徑
DB_PATH = 'music.db'
# 這是我們在 train_kmeans.py 裡儲存的那個 Pipeline 檔案
MODEL_PATH = 'models/music_clustering_pipeline.pkl'

class MusicRecommender:
    def __init__(self):
        self.pipeline = None
        self.df_songs = None
        self.global_knn = None # 用來存每個 Cluster 的 KNN 模型
        
        # 初始化時就先把資料載入，這樣網站跑起來才會快
        self.load_models_and_data()

    def load_models_and_data(self):
        print("   [Recommender] Loading pipeline model...")
        if Path(MODEL_PATH).exists():
            self.pipeline = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        print("   [Recommender] Loading songs from database...")
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT id, id_spotify, title, artist, bpm, energy, danceability, happiness, 
               acousticness, instrumentalness, liveness, speechiness, loudness,
               key, camelot, cluster, url_youtube, url_spotify
        FROM songs
        WHERE cluster IS NOT NULL
        """
        self.df_songs = pd.read_sql(query, conn)
        conn.close()

        # 建立全域 KNN 索引
        self._build_global_knn()

    def _build_global_knn(self):
        """
        建立一個搜尋全資料庫的 KNN 模型
        """
        print("   [Recommender] Building Global KNN index...")
        
        # 取得 Pipeline 中的前處理器
        preprocessor = self.pipeline.named_steps['preprocessor']
        
        # 將所有歌曲轉換成特徵向量
        self.X_all_transformed = preprocessor.transform(self.df_songs)
        
        # 訓練全域 KNN
        self.global_knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
        self.global_knn.fit(self.X_all_transformed)

    def predict_new_song(self, song_data_dict):
        """
        【主功能】
        輸入: 一個單一歌曲的字典 (從網頁表單來的數據)
        輸出: (所屬群組 ID, 推薦歌曲列表)
        """
        # --- [修正點 1] 單位轉換 (0-1 -> 0-100) ---
        # 因為資料庫存的是 0-100，但前端滑桿是 0.0-1.0
        # 我們只轉換那些需要轉的欄位
        scale_100_cols = ['energy', 'danceability', 'happiness', 'acousticness', 
                          'instrumentalness', 'liveness', 'speechiness']
        processed_data = song_data_dict.copy()
        for col in scale_100_cols:
            if col in processed_data:
                processed_data[col] = processed_data[col] * 100

        new_df = pd.DataFrame([processed_data])
        expected_cols = ['bpm', 'energy', 'danceability', 'happiness', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness', 'loudness', 'key', 'camelot']
        for col in expected_cols:
            if col not in new_df.columns:
                new_df[col] = 0 if col not in ['key', 'camelot'] else 'Unknown'

        # 1. 預測 Cluster (還是需要知道它屬於哪一類)
        predicted_cluster = self.pipeline.predict(new_df)[0]
        
        # 2. 全域搜尋相似歌曲 (不再受限於 Cluster)
        recommendations = self._find_global_neighbors(new_df)
        
        return predicted_cluster, recommendations

    def _find_global_neighbors(self, new_df, n_recommendations=10):
        """
        全域搜尋鄰居
        """
        preprocessor = self.pipeline.named_steps['preprocessor']
        new_song_vec = preprocessor.transform(new_df)
        
        # 搜尋最近的 N 個
        distances, neighbor_indices = self.global_knn.kneighbors(new_song_vec, n_neighbors=n_recommendations)
        
        results = []
        for i in range(len(neighbor_indices[0])):
            idx = neighbor_indices[0][i]
            dist = distances[0][i]
            
            song = self.df_songs.iloc[idx]
            
            results.append({
                'title': song['title'],
                'artist': song['artist'],
                'url_youtube': song['url_youtube'] if song['url_youtube'] else '',
                'distance': float(dist),
                'cluster': int(song['cluster']), # 回傳 Cluster 供前端顯示
                'bpm': float(song['bpm']),
                'camelot': song['camelot'] if song['camelot'] else 'Unknown',
                'key': song['key']
            })
            
        return results
    
    def get_pca_scatter_plot(self, new_song_dict=None):
        """
        產生 PCA 降維後的 2D 散佈圖 JSON
        """
        print("   [Visualizer] Generating PCA plot...")
        
        # 1. 準備資料庫歌曲的特徵
        preprocessor = self.pipeline.named_steps['preprocessor']
        X_db = preprocessor.transform(self.df_songs)
        
        # 2. 訓練 PCA (將高維轉成 2D)
        pca = PCA(n_components=2)
        X_embedded = pca.fit_transform(X_db)
        
        # 建立繪圖用的 DataFrame
        plot_df = self.df_songs[['title', 'artist', 'cluster']].copy()
        plot_df['x'] = X_embedded[:, 0]
        plot_df['y'] = X_embedded[:, 1]
        plot_df['Type'] = 'Database Song' # 標記類型
        plot_df['size'] = 5 # 點的大小
        
        # 3. 如果有新歌，將其投影進去
        if new_song_dict:
            # 轉成 DataFrame 並補齊欄位 (同 predict 邏輯)
            new_song_df = pd.DataFrame([new_song_dict])
            expected_cols = ['bpm', 'energy', 'danceability', 'happiness', 'acousticness', 
                             'instrumentalness', 'liveness', 'speechiness', 'loudness', 'key', 'camelot']
            for col in expected_cols:
                if col not in new_song_df.columns:
                    new_song_df[col] = 0 if col not in ['key', 'camelot'] else 'Unknown'
            
            # 轉換特徵 -> PCA 投影
            new_vec = preprocessor.transform(new_song_df)
            new_embedded = pca.transform(new_vec)
            
            # 加入繪圖資料
            new_point = pd.DataFrame({
                'title': ['🔴 YOUR SONG'],
                'artist': ['You'],
                'cluster': [-1], # 特殊顏色
                'x': [new_embedded[0, 0]],
                'y': [new_embedded[0, 1]],
                'Type': ['New Input'],
                'size': [15] # 讓新歌的點大一點
            })
            
            plot_df = pd.concat([plot_df, new_point], ignore_index=True)

        # 4. 使用 Plotly 畫圖
        # 根據 Cluster 上色
        fig = px.scatter(
            plot_df, x='x', y='y', 
            color='cluster', 
            hover_data=['title', 'artist'],
            size='size', # 依大小顯示
            symbol='Type', # 新歌和舊歌用不同形狀
            title='Music Space PCA Projection',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        # 優化圖表外觀
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', # 透明背景
            plot_bgcolor='rgba(240,240,240,0.5)',
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2"
        )

        # 轉成 JSON 讓前端可以讀取
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def get_pca_3d_plot(self, new_song_dict=None, predicted_id=None):
        """
        修正版 3D PCA: 
        1. 接收外部傳入的 predicted_id，確保標題一致
        2. [關鍵] 在內部進行 *100 的單位轉換，確保紅點位置正確
        """
        import plotly.graph_objects as go
        from sklearn.metrics.pairwise import euclidean_distances
        
        print(f"   [Visualizer] Generating 3D Plot for Cluster {predicted_id}...")
        
        # 1. 準備全體資料
        preprocessor = self.pipeline.named_steps['preprocessor']
        X_all_transformed = preprocessor.transform(self.df_songs)
        
        pca = PCA(n_components=3)
        X_embedded_all = pca.fit_transform(X_all_transformed)
        
        plot_df = self.df_songs[['title', 'artist', 'cluster', 'url_spotify', 'url_youtube']].copy()
        plot_df['x'] = X_embedded_all[:, 0]
        plot_df['y'] = X_embedded_all[:, 1]
        plot_df['z'] = X_embedded_all[:, 2]
        plot_df['cluster'] = plot_df['cluster'].fillna(-1).apply(lambda x: str(int(float(x))))
        plot_df['Type'] = 'Database'
        
        centroids = plot_df.groupby('cluster')[['x', 'y', 'z']].mean().reset_index()

        # 2. 處理新歌
        new_row = None
        # 如果沒有傳入 predicted_id，就預設為 None (或轉字串)
        predicted_cluster_str = str(predicted_id) if predicted_id is not None else None
        
        if new_song_dict:
            # --- [關鍵修正] 單位轉換邏輯 (必須與 predict_new_song 一致) ---
            # 複製這份資料以免汙染原始 dict
            processed_data = new_song_dict.copy()
            scale_100_cols = ['energy', 'danceability', 'happiness', 'acousticness', 
                              'instrumentalness', 'liveness', 'speechiness']
            
            for col in scale_100_cols:
                if col in processed_data:
                    # 只有當數值小於等於 1.0 時才乘 (避免重複乘)
                    if isinstance(processed_data[col], (int, float)) and processed_data[col] <= 1.0:
                        processed_data[col] = processed_data[col] * 100

            new_song_df = pd.DataFrame([processed_data])
            # -------------------------------------------------------

            expected_cols = ['bpm', 'energy', 'danceability', 'happiness', 'acousticness', 
                             'instrumentalness', 'liveness', 'speechiness', 'loudness', 'key', 'camelot']
            for col in expected_cols:
                if col not in new_song_df.columns:
                    new_song_df[col] = 0 if col not in ['key', 'camelot'] else 'Unknown'

            new_vec = preprocessor.transform(new_song_df)
            dists = euclidean_distances(new_vec, X_all_transformed)[0]
            plot_df['distance_to_input'] = dists

            new_embedded = pca.transform(new_vec)
            new_x, new_y, new_z = new_embedded[0]
            
            # 這裡我們不再自己 predict 了，直接信任傳進來的 predicted_id
            # 這樣可以保證網頁顯示什麼，圖表標題就是什麼

            new_row = pd.DataFrame({
                'title': ['🔴 YOUR SONG'],
                'artist': ['You'],
                'cluster': ['Input'],
                'x': [new_x], 'y': [new_y], 'z': [new_z],
                'Type': ['Input'],
                'distance_to_input': [0.0]
            })

        # 3. 混合篩選邏輯
        df_global_nearest = plot_df.sort_values('distance_to_input').head(250)
        
        if predicted_cluster_str:
            df_cluster_subset = plot_df[plot_df['cluster'] == predicted_cluster_str]
            if not df_cluster_subset.empty:
                df_cluster_nearest = df_cluster_subset.sort_values('distance_to_input').head(25)
                final_plot_df = pd.concat([df_global_nearest, df_cluster_nearest]).drop_duplicates(subset=['title', 'artist'])
            else:
                final_plot_df = df_global_nearest
        else:
            final_plot_df = df_global_nearest

        # 4. 繪圖 (保持不變)
        fig = go.Figure()

        unique_clusters = sorted(final_plot_df['cluster'].unique(), key=lambda x: int(x) if x.isdigit() else 999)
        for c in unique_clusters:
            cluster_data = final_plot_df[final_plot_df['cluster'] == c]
            fig.add_trace(go.Scatter3d(
                x=cluster_data['x'], y=cluster_data['y'], z=cluster_data['z'],
                mode='markers',
                name=f'Cluster {c}',
                text=cluster_data.apply(lambda row: f"<b>{row['title']}</b><br>{row['artist']}<br>Dist: {row['distance_to_input']:.2f}", axis=1),
                hoverinfo='text',
                marker=dict(size=4, opacity=0.6)
            ))

        visible_centroids = centroids[centroids['cluster'].isin(unique_clusters)]
        fig.add_trace(go.Scatter3d(
            x=visible_centroids['x'], y=visible_centroids['y'], z=visible_centroids['z'],
            mode='markers+text',
            name='Cluster Centers',
            text=visible_centroids['cluster'].apply(lambda c: f"C{c}"),
            textposition="top center",
            hoverinfo='none',
            marker=dict(size=8, symbol='diamond', color='black', opacity=0.5)
        ))

        if new_row is not None:
            fig.add_trace(go.Scatter3d(
                x=new_row['x'], y=new_row['y'], z=new_row['z'],
                mode='markers+text',
                name='YOUR SONG',
                text=["YOUR SONG"],
                textposition="top center",
                hoverinfo='none',
                marker=dict(size=6, color='red', symbol='circle', line=dict(color='white', width=1))
            ))
            
            nearest_5 = final_plot_df.sort_values('distance_to_input').head(5)
            for _, row in nearest_5.iterrows():
                fig.add_trace(go.Scatter3d(
                    x=[new_row['x'][0], row['x']],
                    y=[new_row['y'][0], row['y']],
                    z=[new_row['z'][0], row['z']],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    hoverinfo='none',
                    showlegend=False
                ))

        # 標題使用傳進來的 ID
        title_str = f'Music Space (Pred: C{predicted_cluster_str})' if predicted_cluster_str else 'Music Space'
        
        fig.update_layout(
            title=title_str,
            paper_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3',
                bgcolor='rgba(240,240,240,0.5)'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def get_polar_plot(self, recommendations, input_camelot='8B'):
        """
        畫出 Camelot Wheel (五度圈) 雷達圖
        角度 = Camelot Key (1-12 點鐘方向)
        半徑 = 距離 (越近越像)
        顏色 = Cluster
        """
        print("   [Visualizer] Generating Camelot Polar plot...")
        
        # Helper: 解析 Camelot (例如 "8B" -> 角度)
        def get_camelot_angle(cam_str):
            if not cam_str or cam_str == 'Unknown':
                return 0
            # 抓取數字部分
            match = re.match(r"(\d+)([AB]?)", str(cam_str))
            if match:
                num = int(match.group(1))
                letter = match.group(2)
                # 12點鐘 = 12 * 30度 = 360 (或 0)
                # 每個小時 30 度
                base_angle = num * 30 
                # 為了區分 A (Minor) 和 B (Major)，可以稍微錯開一點
                # 例如 B (Major) 在正點，A (Minor) 往內縮或偏移 10 度
                # 這裡簡單處理：B 放在正刻度，A 偏移 15 度
                offset = 0 if letter == 'B' else 15
                return (base_angle + offset) % 360
            return 0

        titles = []
        dists = []
        angles = [] 
        hover_texts = []
        colors = []
        symbols = []
        sizes = []

        # 1. 中心點 (使用者)
        titles.append("YOUR SONG")
        dists.append(0) 
        # 假設使用者有輸入 Camelot，沒有就預設
        angles.append(get_camelot_angle(input_camelot)) 
        hover_texts.append(f"<b>YOUR SONG</b><br>Key: {input_camelot}")
        colors.append('#FF0000') # 紅色
        symbols.append('star')
        sizes.append(25)

        # 2. 鄰居點
        # 定義 Cluster 的顏色表 (離散顏色)
        cluster_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78'
        ]

        for song in recommendations:
            titles.append(song['title'])
            dists.append(song['distance'])
            
            # 計算角度 (Key)
            angle = get_camelot_angle(song['camelot'])
            angles.append(angle)
            
            # 顏色根據 Cluster
            c_idx = song['cluster'] % len(cluster_colors)
            colors.append(cluster_colors[c_idx])
            
            hover_text = (f"<b>{song['title']}</b><br>{song['artist']}<br>"
                          f"Key: {song['camelot']} ({song['key']})<br>"
                          f"Cluster: {song['cluster']}<br>"
                          f"Dist: {song['distance']:.2f}")
            hover_texts.append(hover_text)
            symbols.append('circle')
            sizes.append(12)

        fig = go.Figure()

        # 畫點
        fig.add_trace(go.Scatterpolar(
            r=dists,
            theta=angles,
            mode='markers',
            text=titles,
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(
                color=colors,
                size=sizes,
                symbol=symbols,
                line=dict(color='white', width=1),
                opacity=0.8
            )
        ))

        fig.update_layout(
            title='音樂調性雷達 (Camelot Wheel)',
            polar=dict(
                radialaxis=dict(visible=True, title='距離 (越近越像)', range=[0, max(dists)*1.2]),
                angularaxis=dict(
                    tickmode='array',
                    # 顯示 1B 到 12B 的刻度
                    tickvals=[i*30 for i in range(1, 13)],
                    ticktext=[f"{i}B/A" for i in range(1, 13)],
                    direction="clockwise", # 順時針比較符合 DJ 習慣
                    rotation=90 # 12點鐘在上方
                )
            ),
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # 加上註釋：解釋顏色
        fig.add_annotation(
            text="顏色代表不同 Cluster，角度代表調性 (相鄰可混音)",
            xref="paper", yref="paper",
            x=0.5, y=-0.1, showarrow=False
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ==========================================
# 測試區 (可以直接執行這個檔案來測試功能)
# ==========================================
if __name__ == "__main__":
    try:
        recommender = MusicRecommender()
        
        # 模擬一首從網頁傳過來的歌
        mock_song = {
            'bpm': 120, 
            'energy': 0.8, 
            'danceability': 0.7, 
            'happiness': 0.6,
            'acousticness': 0.1, 
            'instrumentalness': 0, 
            'liveness': 0.2, 
            'speechiness': 0.05, 
            'loudness': -5,
            'key': 'C Major', 
            'camelot': '8B'
        }
        
        print(f"\n🧪 Testing with song: {mock_song}")
        cluster, recs = recommender.predict_new_song(mock_song)
        
        print(f"🔮 Predicted Cluster: {cluster}")
        print("🎶 Recommended Songs:")
        for song in recs:
            print(f"   - {song['title']} ({song['artist']}) [Dist: {song['distance']:.2f}]")
            
    except Exception as e:
        print(f"❌ Error: {e}")