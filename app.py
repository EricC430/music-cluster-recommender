from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 引用我們之前寫好的模組
from recommender import MusicRecommender
from scraper import Chosic_Song_Analysis_Scraper, parse_html_data

app = Flask(__name__)

CAMELOT_TO_KEY = {
    "1B": "B Major", "2B": "F# Major", "3B": "Db Major", "4B": "Ab Major", 
    "5B": "Eb Major", "6B": "Bb Major", "7B": "F Major", "8B": "C Major", 
    "9B": "G Major", "10B": "D Major", "11B": "A Major", "12B": "E Major",
    "1A": "Ab Minor", "2A": "Eb Minor", "3A": "Bb Minor", "4A": "F Minor", 
    "5A": "C Minor", "6A": "G Minor", "7A": "D Minor", "8A": "A Minor", 
    "9A": "E Minor", "10A": "B Minor", "11A": "F# Minor", "12A": "Db Minor",
    "Unknown": "Unknown"
}

# --- 初始化區塊 ---
print("⏳ 初始化推薦系統...")
recommender = MusicRecommender()
print("✅ 推薦系統載入完成")

if not os.path.exists('static'):
    os.makedirs('static')

# 確保圖片存在
tree_img_src = 'models/cluster_tree_plot.png'
tree_img_dst = 'static/cluster_tree_plot.png'
if os.path.exists(tree_img_src):
    shutil.copy(tree_img_src, tree_img_dst)

# --- 路由設定 ---

@app.route('/')
def index():
    return render_template('index.html')

# 【新增】爬蟲 API
@app.route('/api/import_spotify', methods=['POST'])
def import_spotify():
    """
    接收 JSON 格式的 { "spotify_id": "..." }
    啟動爬蟲 -> 解析 -> 回傳特徵 JSON
    """
    data = request.get_json()
    song_id = data.get('spotify_id', '').strip()

    if not song_id:
        return jsonify({"error": "請輸入 Spotify ID"}), 400

    print(f"🕷️ 收到爬取請求，ID: {song_id}，正在啟動 Headless Driver...")

    # 啟動隱形瀏覽器 (Headless) 以加快速度
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new") # 隱形模式
    options.add_argument("--disable-notifications")
    options.add_argument("--window-size=1920,1080")
    # 偽裝 User-Agent
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        # 1. 爬取
        scraper = Chosic_Song_Analysis_Scraper(driver, song_id)
        html = scraper.scrape() # 這會回傳 HTML string

        if not html:
            return jsonify({"error": "爬取失敗，Chosic 網站無回應或 ID 錯誤"}), 500

        # 2. 解析
        # parser.py 的 parse_html_data 回傳的是一個字典
        clean_data = parse_html_data(html, song_id)
        
        # 檢查是否成功抓到數值 (簡單檢查 BPM)
        if clean_data.get('bpm', 0) == 0:
             return jsonify({"error": "分析失敗 (資料不完整)"}), 500

        print(f"✅ 爬取成功: {clean_data['title']} - {clean_data['artist']}")
        
        # 3. 回傳 JSON 給前端
        return jsonify({
            "success": True,
            "data": clean_data
        })

    except Exception as e:
        print(f"❌ API 錯誤: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        driver.quit()

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # 1. 獲取使用者選的 Camelot 代碼 (例如 "8B")
        selected_camelot = request.form.get('camelot_code', 'Unknown')
        
        # 2. 自動對應 Key (後端處理)
        derived_key = CAMELOT_TO_KEY.get(selected_camelot, "Unknown")

        # 獲取所有特徵 (包含全部 9 個數值特徵)
        features = {
            'bpm': float(request.form.get('bpm', 120)),
            'energy': float(request.form.get('energy', 0.5)),
            'danceability': float(request.form.get('danceability', 0.5)),
            'happiness': float(request.form.get('happiness', 0.5)),
            'acousticness': float(request.form.get('acousticness', 0.1)),
            'instrumentalness': float(request.form.get('instrumentalness', 0.0)),
            'liveness': float(request.form.get('liveness', 0.1)), 
            'speechiness': float(request.form.get('speechiness', 0.1)),
            'loudness': float(request.form.get('loudness', -10)),
            'camelot': selected_camelot,
            'key': derived_key
        }

        # 1. 先預測，拿到正確的 cluster_id (這裡是 12)
        cluster_id, recommendations = recommender.predict_new_song(features)
        
        # 2. 畫圖時，把 features 和 cluster_id 都傳進去
        # (A) 3D PCA 圖
        plot_json = recommender.get_pca_3d_plot(features, predicted_id=cluster_id)
        
        # (B) 雷達圖
        input_camelot = features.get('camelot', '8B')
        polar_json = recommender.get_polar_plot(recommendations, input_camelot)
        
        return render_template(
            'result.html', 
            cluster_id=cluster_id, 
            features=features,
            recommendations=recommendations,
            plot_json=plot_json,
            polar_json=polar_json
        )

    except Exception as e:
        return f"❌ 系統發生錯誤: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)