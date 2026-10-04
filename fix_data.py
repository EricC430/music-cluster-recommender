import sqlite3
import time
import random
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 引用 scraper 的功能
from scraper import Chosic_Song_Analysis_Scraper, parse_html_data

DB_NAME = "music.db"

class DataFixer:
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME)
        self.conn.row_factory = sqlite3.Row
        self.driver = None

        # ==========================================
        # 【核心配置】修復策略清單
        # 每個策略包含:
        # 1. name: 策略名稱 (Log用)
        # 2. condition: SQL WHERE 子句，找出有問題的資料
        # 3. handler: 對應的修復函式
        # ==========================================
        self.fix_strategies = [
            {
                "name": "修復日期格式 (本地優先)",
                # 條件: 日期不是 YYYY-MM-DD 格式，且不是空的 (空的無法轉，直接去重爬)
                "condition": "release_date NOT LIKE '____-__-__' AND release_date IS NOT NULL AND release_date != ''",
                "handler": self.strategy_local_date_fix_then_scrape
            },
            {
                "name": "標記空 Wiki Genres 為 N/A (本地)",
                # 條件: 只有當 Wiki 是空的，但其他資料看起來是正常的 (例如 duration 有值) 才補 N/A
                # 這樣可以避免把那種「完全沒爬成功(duration=0)」的資料也誤標為 N/A
                "condition": "(genres_wikipedia = '' OR genres_wikipedia IS NULL) AND duration > 0",
                "handler": self.strategy_mark_wiki_na
            },
            {
                "name": "清洗 Album 前綴贅字 (本地)",
                # 條件: 專輯名稱開頭是 "from " 且中間包含 "album:" 的資料
                "condition": "album LIKE 'from %album:%'",
                "handler": self.strategy_clean_album_prefix
            },
            {
                "name": "補抓遺失資料 (重爬)",
                # 條件: Duration 為 0 或 NULL，或是日期、Wiki Genre 是空的
                "condition": "duration = 0 OR duration IS NULL OR release_date = '' OR genres_wikipedia = ''",
                "handler": self.strategy_full_rescrape
            }
        ]

    def init_driver(self):
        """延遲初始化 Driver，只有在需要爬蟲時才開啟"""
        if self.driver is None:
            print("   🔌 正在啟動瀏覽器...")
            options = webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            options.add_argument("--disable-notifications")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            options.add_argument(f'user-agent={user_agent}')
            
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # ==========================================
    # 工具函式
    # ==========================================
    def convert_date_string(self, date_str):
        """嘗試將各種日期格式轉為 YYYY-MM-DD"""
        try:
            # 格式 1: October 19, 2018
            dt = datetime.strptime(date_str.strip(), "%B %d, %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
            
        try:
            # 格式 2: 2018 (僅年份) -> 2018-01-01
            dt = datetime.strptime(date_str.strip(), "%Y")
            return dt.strftime("%Y-01-01")
        except ValueError:
            pass

        return None # 轉換失敗

    def update_single_field(self, song_id, field, value):
        """快速更新單一欄位 (用於本地修復)"""
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE songs SET {field} = ? WHERE id_spotify = ?", (value, song_id))
        self.conn.commit()

    def update_full_record(self, data):
        """更新整筆資料 (用於爬蟲回填)"""
        cursor = self.conn.cursor()
        update_sql = """
        UPDATE songs SET 
            title = :title, artist = :artist, album = :album, release_date = :release_date,
            bpm = :bpm, duration = :duration, key = :key, camelot = :camelot,
            popularity = :popularity, energy = :energy, danceability = :danceability,
            happiness = :happiness, acousticness = :acousticness, instrumentalness = :instrumentalness,
            liveness = :liveness, speechiness = :speechiness, loudness = :loudness,
            genres_spotify = :genres_spotify, genres_wikipedia = :genres_wikipedia,
            url_youtube = :url_youtube
        WHERE id_spotify = :id
        """
        try:
            cursor.execute(update_sql, data)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"   ❌ 更新 DB 失敗: {e}")
            return False

    # ==========================================
    # 策略實作 (Handlers)
    # ==========================================

    def strategy_local_date_fix_then_scrape(self, row):
        """
        策略 A: 日期修復
        流程: 1. 嘗試 Python 字串轉換 -> 2. 成功則 Update DB -> 3. 失敗則呼叫爬蟲
        """
        song_id = row['id_spotify']
        raw_date = row['release_date']
        
        print(f"   [處理中] ID: {song_id} | 原始日期: '{raw_date}'")

        # 1. 嘗試本地轉換
        new_date = self.convert_date_string(raw_date)
        
        if new_date:
            print(f"   ✅ 本地轉換成功! {raw_date} -> {new_date}")
            self.update_single_field(song_id, "release_date", new_date)
            return True # 修復完成
        else:
            print(f"   ⚠️ 本地轉換失敗，轉交爬蟲重抓...")
            # 2. 本地失敗，轉交給爬蟲處理
            return self.strategy_full_rescrape(row)

    def strategy_full_rescrape(self, row):
        """
        策略 B: 完整重爬
        流程: 1. 啟動 Driver -> 2. 爬取 -> 3. 解析 -> 4. 更新 DB
        """
        self.init_driver() # 確保瀏覽器已啟動
        song_id = row['id_spotify']
        
        try:
            # 1. 爬取
            scraper = Chosic_Song_Analysis_Scraper(self.driver, song_id)
            html = scraper.scrape()
            scraper.random_sleep(2,3)
            
            if html:
                # 2. 解析 (parser 已經包含新的日期處理邏輯)
                clean_data = parse_html_data(html, song_id)
                clean_data['id'] = song_id # 補上 ID 供 SQL 使用
                
                # 3. 檢查是否有效 (例如 Duration > 0)
                if clean_data.get('duration', 0) > 0:
                    self.update_full_record(clean_data)
                    print(f"   ✅ 重爬並更新成功: {clean_data['title']}")
                    return True
                else:
                    print(f"   ⚠️ 重爬後資料無效 (Duration=0)")
            else:
                print(f"   ⚠️ 網頁無回應")
            
            # 避免被擋
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"   ❌ 重爬發生錯誤: {e}")
        
        return False

    def strategy_mark_wiki_na(self, row):
        """
        策略 C: 本地強制標記 Wiki Genre 為 N/A
        用途: 針對那些確定沒有資料，或者是想要先補值避免重複爬取的資料
        """
        song_id = row['id_spotify']
        print(f"   [本地修正] ID: {song_id} | Wiki Genres: '' -> 'N/A'")
        
        # 使用你原本寫好的 update_single_field 快速更新
        self.update_single_field(song_id, "genres_wikipedia", "N/A")
        return True
    
    def strategy_clean_album_prefix(self, row):
        """
        策略 D: 本地清洗專輯名稱前綴
        將 "from compilation album: ABC" 改為 "ABC"
        """
        song_id = row['id_spotify']
        original_album = row['album']
        
        # 使用與 scraper 相同的 Regex 邏輯進行清洗
        # 這裡多加一個判斷，如果清完變空的（雖然機率低），就保持原樣或設 Unknown
        clean_album = re.sub(r'^from\s+.*?\s*album:\s*', '', original_album, flags=re.IGNORECASE).strip()
        
        if clean_album != original_album:
            print(f"   [Album清洗] ID: {song_id}")
            print(f"      └─ '{original_album}' -> '{clean_album}'")
            self.update_single_field(song_id, "album", clean_album)
            return True
        return False

    # ==========================================
    # 主流程
    # ==========================================
    def run(self):
        print("🚀 開始執行資料庫修復系統...")
        
        cursor = self.conn.cursor()

        # 迭代每個策略
        for strategy in self.fix_strategies:
            print("\n" + "="*50)
            print(f"🛠️  執行策略: {strategy['name']}")
            print(f"    條件: {strategy['condition']}")
            print("="*50)
            
            # 1. 找出符合該策略條件的資料
            query = f"SELECT * FROM songs WHERE {strategy['condition']}"
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                print("    ✨ 沒有符合此條件的錯誤資料。")
                continue
                
            print(f"    🔍 發現 {len(rows)} 筆資料待處理...")
            
            # 2. 逐筆執行對應的 handler
            success_count = 0
            for row in rows:
                # 執行綁定的函式 (handler)
                result = strategy['handler'](row)
                if result:
                    success_count += 1
            
            print(f"    🏁 策略結束. 成功修復: {success_count}/{len(rows)}")

        print("\n🎉 所有修復作業完成！")
        if self.driver:
            self.driver.quit()
        self.conn.close()

if __name__ == "__main__":
    fixer = DataFixer()
    fixer.run()