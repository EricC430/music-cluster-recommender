import sqlite3
import os

DB_NAME = "music.db"

def init_db():
    """
    初始化資料庫
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. 修改 Schema: 
    #    - 新增 "id_spotify" TEXT UNIQUE
    #    - "id" 維持 INTEGER PRIMARY KEY AUTOINCREMENT
    sql_create_table = """
    CREATE TABLE IF NOT EXISTS songs (
        "id" INTEGER PRIMARY KEY AUTOINCREMENT,
        "id_spotify" TEXT UNIQUE,  
        "title" TEXT,
        "artist" TEXT,
        "key" TEXT,
        "camelot" TEXT,
        "bpm" INTEGER,
        "duration" INTEGER,
        "release_date" TEXT,
        "album" TEXT,
        "popularity" INTEGER,
        "energy" INTEGER,
        "danceability" INTEGER,
        "happiness" INTEGER,
        "acousticness" INTEGER,
        "instrumentalness" INTEGER,
        "liveness" INTEGER,
        "speechiness" INTEGER,
        "loudness" INTEGER,
        "genres_spotify" TEXT,
        "genres_wikipedia" TEXT,
        "cluster" INTEGER,
        "url_spotify" TEXT,
        "url_youtube" TEXT
    );
    """
    c.execute(sql_create_table)
    conn.commit()
    conn.close()
    print("✅ 資料庫表單已更新 (id_spotify UNIQUE)")

def save_song(data):
    """
    將解析後的資料寫入資料庫
    """
    if not data:
        return

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    try:
        # 2. 修改 INSERT 語法:
        #    - 使用 "INSERT OR IGNORE": 如果 id_spotify 重複，資料庫會直接忽略這筆，不會報錯，也不會產生重複資料。
        #    - 對應欄位: 將 data['id'] 寫入 "id_spotify" 欄位
        
        insert_sql = """
        INSERT OR IGNORE INTO songs (
            "id_spotify", "title", "artist", "key", "camelot", "bpm", "duration", 
            "release_date", "album", "popularity", "energy", 
            "danceability", "happiness", "acousticness", "instrumentalness", 
            "liveness", "speechiness", "loudness", 
            "genres_spotify", "genres_wikipedia", 
            "url_spotify", "url_youtube"
        ) VALUES (
            :id, :title, :artist, :key, :camelot, :bpm, :duration, 
            :release_date, :album, :popularity, :energy, 
            :danceability, :happiness, :acousticness, :instrumentalness, 
            :liveness, :speechiness, :loudness, 
            :genres_spotify, :genres_wikipedia, 
            :url_spotify, :url_youtube
        )
        """
        
        # 這裡的 data 是字典，:id 會去抓 data['id'] 的值
        c.execute(insert_sql, data)
        
        # 檢查是否有實際寫入資料 (rowcount > 0 代表成功寫入，0 代表重複被忽略)
        if c.rowcount > 0:
            conn.commit()
            print(f"💾 已儲存: {data['title']}")
        else:
            print(f"⚠️ 跳過重複 (id_spotify 已存在): {data['title']}")

    except Exception as e:
        print(f"❌ 資料庫寫入錯誤: {e}")
    finally:
        conn.close()