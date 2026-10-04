# Music Cluster Recommender｜音樂分群推薦系統

以 **K-Means 分群 + K 近鄰（KNN）** 為核心的音樂推薦網站。系統從 [Chosic](https://www.chosic.com/) 爬取歌曲的音訊特徵（BPM、能量、舞動性、調性等），存入 SQLite，訓練分群模型後，讓使用者輸入一首歌的特徵（或貼上 Spotify 曲目 ID 自動抓取），即可得到：

- 這首歌所屬的**群組（cluster）**
- 資料庫中最相似的 **10 首推薦歌曲**（含 YouTube 連結）
- **3D PCA 音樂空間圖**：看見你的歌落在哪、鄰居有誰
- **Camelot Wheel 調性雷達圖**：依調性與距離呈現可混音（DJ mixing）的歌曲
- **決策樹代理模型**：用淺層決策樹解釋各群組的特徵規則

> 本專案為《網路與資料庫程式設計》期末專題。

## 系統架構

```
Chosic 網站
   │  Selenium + BeautifulSoup
   ▼
scraper.py ──► db_manager.py ──► music.db (SQLite, songs 表)
                                     │
                       fix_data.py ──┤ 修復缺漏 / 格式錯誤的資料
                                     ▼
              tune_kmeans.py（挑 K）→ train_kmeans.py
                                     │  StandardScaler + OneHotEncoder + KMeans (Pipeline)
                                     ▼
                  models/music_clustering_pipeline.pkl  ＋ 寫回 songs.cluster
                                     │
        interpret_model.py（決策樹解釋分群）
                                     ▼
              recommender.py (MusicRecommender) ◄── app.py (Flask)
                                                       ├─ GET  /                  輸入表單
                                                       ├─ POST /api/import_spotify  即時爬取單曲特徵
                                                       └─ POST /recommend          預測 + 推薦 + 視覺化
```

重點設計：

- **訓練與推論共用同一個 sklearn `Pipeline`**，新歌一定經過與訓練時相同的標準化／編碼。
- 推薦是對**整個資料庫**做全域 KNN（不限於預測的群組），群組僅用於解釋與視覺化。
- 資料庫中的百分比特徵（energy、danceability…）為 0–100，網頁滑桿為 0–1，後端在 `recommender.py` 內統一換算。

## 技術棧

Python 3.10 · Flask · SQLite · scikit-learn · pandas · Plotly · Selenium · BeautifulSoup · Bootstrap 5

## 快速開始

> ⚠️ 倉庫**不含**爬取的資料（`music.db`）與訓練好的模型（`.pkl`），請依下列步驟自行產生。

### 1. 安裝

需要 Python 3.10+ 與本機安裝的 Google Chrome（ChromeDriver 由 `webdriver-manager` 自動下載）。

```bash
git clone https://github.com/EricC430/music-cluster-recommender.git
cd music-cluster-recommender
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 建立資料與模型

```bash
python scraper.py        # 建立 music.db 並爬取歌曲（耗時，可隨時中斷，會自動跳過已存在的歌曲）
python fix_data.py       # （選用）修復日期格式、補抓缺漏欄位
python tune_kmeans.py    # （選用）以 Elbow / Silhouette 圖挑選 K
python train_kmeans.py   # 訓練分群模型、存成 models/*.pkl 並把 cluster 寫回資料庫
python interpret_model.py  # （選用）產生決策樹解釋圖 models/cluster_tree_plot.png
```

`scraper.py` 內可調整 `START_TOPIC_INDEX`（從第幾個主題開始爬）與 `LOG_MODE`（是否寫 JSON log）。`train_kmeans.py` 預設 `n_clusters=16`，可依 `tune_kmeans.py` 的結果修改。

### 3. 啟動網站

```bash
python app.py
```

開啟 <http://127.0.0.1:5000>，調整滑桿或輸入 Spotify 曲目 ID 後送出即可。

## 專案結構

| 檔案 | 說明 |
| --- | --- |
| `app.py` | Flask 入口與路由、Camelot ↔ 調性對照 |
| `recommender.py` | `MusicRecommender`：載入模型、全域 KNN、3D PCA 圖、Camelot 雷達圖 |
| `scraper.py` | Chosic 爬蟲（主題清單 / 歌單 / 單曲分析）與 HTML 解析 |
| `db_manager.py` | SQLite schema 與 `save_song()`（以 `id_spotify` 去重） |
| `fix_data.py` | 資料修復策略（本地修正優先，必要時重爬） |
| `train_kmeans.py` | 分群模型訓練與寫回資料庫 |
| `tune_kmeans.py` | K 值探索（StandardScaler vs PowerTransformer） |
| `interpret_model.py` | 決策樹代理模型，解釋分群規則 |
| `templates/` | `index.html`（輸入頁）、`result.html`（結果頁） |
| `models/` | 訓練產物（`.pkl` 不入版控）與決策樹圖 |

## 注意事項

- 爬蟲僅供學習與研究用途，請遵守 Chosic 網站的使用條款並維持合理的請求頻率（程式內已加入隨機延遲）。
- 網頁抓取依賴 Chosic 的頁面結構，網站改版可能導致解析失敗。
- `/api/import_spotify` 與 `/recommend` 目前為開發用途（Flask `debug=True`），未針對正式部署做強化。
