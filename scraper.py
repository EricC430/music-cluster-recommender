import time
import random
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    ElementClickInterceptedException,
    StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager
from db_manager import init_db, save_song 

LOG_MODE = 0
START_TOPIC_INDEX = 0

def save_to_db(data_list):
    pass

class Scraper:
    def __init__(self, driver):
        self.root_folder = Path("./logs")
        self.root_folder.mkdir(parents=True,exist_ok=True)
        if LOG_MODE:
            self.log_file = self.root_folder / f"log_{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if not self.log_file.exists():
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
        
        self.driver = driver
        self.wait = WebDriverWait(self.driver, 10)

    def log_data(self, data):
        if LOG_MODE:
            try:
                if self.log_file.exists():
                    with open(self.log_file, 'r', encoding='utf-8') as f:
                        try:
                            current_data = json.load(f)
                            if not isinstance(current_data, list): current_data = []
                        except json.JSONDecodeError:
                            current_data = []
                else:
                    current_data = []

                if isinstance(data, list):
                    current_data.extend(data)
                else:
                    current_data.append(data)
                
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    json.dump(current_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"[Error] Logging: {e}")

    def random_sleep(self, min_seconds=1.5, max_seconds=3):
        time.sleep(random.uniform(min_seconds, max_seconds))

    def handle_ads(self):
        ad_close_selectors = ["div#dismiss-button"]

        for selector in ad_close_selectors:
            try:
                close_btn = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                close_btn.click()
                self.random_sleep(1,2)
                return
            except (TimeoutException, NoSuchElementException):
                continue
            except Exception as e:
                print(f"[Error] Ads: {e}")

        try:
            iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
            for iframe in iframes:
                if "google" in iframe.get_attribute("src") or "ad" in iframe.get_attribute("id"):
                    self.driver.find_element(By.TAG_NAME, 'body').click()
                    break
        except:
            pass

    def safe_click(self, element):
        max_retries = 3
        for i in range(max_retries):
            try:
                element.click()
                return True
            except (ElementClickInterceptedException, StaleElementReferenceException):
                self.handle_ads()
                self.random_sleep(1,2)
            except Exception as e:
                self.driver.execute_script("arguments[0].click();", element)
                return True
        return False
    

class Chosic_Topics_Scraper(Scraper):
    def __init__(self, driver):
        super().__init__(driver)
        self.target_url = "https://www.chosic.com/lists-of-songs/"

    def scrape(self):
        self.driver.get(self.target_url)
        self.random_sleep()
        topic_links = []
        try:
            selectors = "div.entry-content a" # "div#topic a, div#bpm a, div#decade a div#word a"
            elements = self.driver.find_elements(By.CSS_SELECTOR, selectors)
            for el in elements:
                href = el.get_attribute("href")
                if href and href.startswith("http") and href not in topic_links:
                    topic_links.append(href)
            self.log_data(topic_links)
            return topic_links
        except Exception as e:
            print(f"[Error] Scraping Links: {e}")
            return []
        
class Chosic_Songlist_Scraper(Scraper):
    def __init__(self, driver, href):
        super().__init__(driver)
        self.target_url = href

    def get_existing_ids(self):
        """
        從資料庫讀取所有已存在的 id_spotify
        """
        existing_ids = set()
        try:
            conn = sqlite3.connect("music.db") # 請確認資料庫名稱是否正確
            cursor = conn.cursor()
            # 確保只抓取 id_spotify 欄位
            cursor.execute("SELECT id_spotify FROM songs")
            rows = cursor.fetchall()
            for row in rows:
                if row[0]: # 確保不是 None
                    existing_ids.add(row[0])
            conn.close()
            print(f"📊 目前資料庫已有 {len(existing_ids)} 筆資料，將自動過濾。")
        except Exception as e:
            print(f"[Warning] 無法讀取資料庫 (如果是第一次執行可忽略): {e}")
        
        return existing_ids

    def scrape(self):
        self.driver.get(self.target_url)
        self.random_sleep()
        self.handle_ads()

        # 滾動以觸發 Lazy Loading 先省略
        existing_ids_set = self.get_existing_ids()

        song_ids = []
        try:
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "track-list-item")))
            items = self.driver.find_elements(By.CSS_SELECTOR, "div.song-div div.track-list-item")
            skip_count = 0
            for item in items:
                data_id = item.get_attribute("data-id")
                if data_id and data_id not in song_ids:
                    if data_id not in existing_ids_set:
                        song_ids.append(data_id)
                    else:
                        skip_count += 1
            
            print(f"✅ 抓取完成: 新增 {len(song_ids)} 筆 (已過濾 {skip_count} 筆重複)")
            self.log_data(song_ids)
            return song_ids
        except Exception as e:
            print(f"[Error] Scraping SongIDs: {e}")
            return []
        
class Chosic_Song_Analysis_Scraper(Scraper):
    def __init__(self, driver, song_id):
        super().__init__(driver)
        self.song_id = song_id
        self.target_url = f"https://www.chosic.com/music-genre-finder/?track={song_id}"

    def scrape(self):
        self.driver.get(self.target_url)
        self.random_sleep()
        self.handle_ads()

        try:
            try:
                search_btn = self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "btn-search")))
                self.safe_click(search_btn)
            except TimeoutException:
                print("[Fail] Search button not found")
            self.wait.until(EC.presence_of_element_located((By.ID, "result-analyzer")))
            
            try:
                yt_btn_locator = (By.CSS_SELECTOR, "i.yt-link")
                yt_btn = self.wait.until(EC.element_to_be_clickable(yt_btn_locator))
                clicked = self.safe_click(yt_btn)
            except TimeoutException:
                print("[Fail] YT button not found")
            # if clicked:
            #     iframe_element = self.wait.until(EC.presence_of_element_located((By.ID, "yt-iframe")))

            self.random_sleep(2,3)

            # # <iframe id="yt-iframe">
            # youtube_url = "N/A"
            # if clicked:
            #     youtube_url = iframe_element.get_attribute("src")
            
            # <article id="result-analyzer">
            result_analyzer_html = "N/A"
            try:
                result_analyzer = self.driver.find_element(By.ID, "result-analyzer")
                result_analyzer_html = result_analyzer.get_attribute('outerHTML')
            except:
                print("[Fail] result-analyzer Not Found")
            
            # <article id="result-p">
            result_p_html = "N/A"
            try:
                result_p = self.driver.find_element(By.ID, "result-p")
                result_p_html = result_p.get_attribute('outerHTML')
            except:
                print("[Fail] result-p Not Found")
            
            full_html = result_p_html + result_analyzer_html
            self.log_data({"result-p": result_p_html, "result-analyzer": result_analyzer_html})
            return full_html
        
        except Exception as e:
            print(f"[Error] Scraping SongIDs: {e}")
            return None

def convert_date_format(date_str):
    """
    將 "October 19, 2018" 格式轉換為 "2018-10-19"
    如果解析失敗，回傳原始字串
    """
    try:
        # 嘗試標準完整格式: Month DD, YYYY
        dt = datetime.strptime(date_str.strip(), "%B %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        try:
            # 嘗試僅有年份: YYYY
            dt = datetime.strptime(date_str.strip(), "%Y")
            return dt.strftime("%Y-01-01") # 或保留僅年份
        except ValueError:
            return date_str

def parse_html_data(html_content, song_id_from_url):
    """
    解析 HTML 並清理 URL 格式，同時支援多位歌手提取。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {}

    # ==========================================
    # 1. 解析基本資訊 (result-p)
    # ==========================================
    p_section = soup.find('article', id='result-p')
    
    if p_section:
        # --- ID ---
        # 優先從 HTML 的 data-id 抓，因為這最準確
        track_item = p_section.find('div', class_='track-list-item')
        if track_item and track_item.has_attr('data-id'):
            data['id'] = track_item['data-id']
        else:
            data['id'] = song_id_from_url

        # --- 歌名 ---
        title_div = p_section.find('div', class_='ng-binding')
        data['title'] = title_div.get_text(strip=True) if title_div else "Unknown"

        # --- 歌手 (支援多位) ---
        # HTML 結構通常是: <div class="track-list-item-info-genres"> <p> by: <a>Artist 1</a>, <a>Artist 2</a> </p> </div>
        artist_div = p_section.find('div', class_='track-list-item-info-genres')
        if artist_div:
            # 抓取裡面所有的 <a> 標籤
            artist_links = artist_div.find_all('a')
            if artist_links:
                # 將多位歌手用 ", " 連接，例如 "Jay Chou, Jolin Tsai"
                data['artist'] = ", ".join([a.get_text(strip=True) for a in artist_links])
            else:
                # 萬一沒有連結，嘗試抓純文字並移除 "by:"
                raw_text = artist_div.get_text(strip=True)
                data['artist'] = raw_text.replace('by:', '').strip()
        else:
            data['artist'] = "Unknown"

        # --- 專輯與發行日 ---
        album_p = p_section.find('p', class_='album-data')
        if album_p:
            raw_text = album_p.get_text(" ", strip=True)
            
            clean_prefix_text = re.sub(r'^from\s+.*?\s*album:\s*', '', raw_text, flags=re.IGNORECASE).strip()
            match = re.search(r'(.*)\s+\(([^)]+)\)$', clean_prefix_text)
            
            if match:
                data['album'] = match.group(1).strip()
                raw_date = match.group(2).strip()
                data['release_date'] = convert_date_format(raw_date)
            else:
                # 萬一 Regex 沒對到 (例如沒有日期)，就整串當專輯
                data['album'] = raw_text
                data['release_date'] = ""
        else:
            data['album'] = "Unknown"
            data['release_date'] = ""

        # --- YouTube URL 清理 ---
        # 目標格式: https://www.youtube.com/watch?v=Bcus42ihkTI
        yt_iframe = p_section.find('iframe', id='yt-iframe')
        data['url_youtube'] = "" # 預設空字串
        
        if yt_iframe and yt_iframe.has_attr('src'):
            src = yt_iframe['src']
            # 使用 Regex 提取 embed/ 後面的 ID (遇到 ? 或結尾停止)
            # src 範例: https://www.youtube.com/embed/Bcus42ihkTI?enablejsapi=1...
            vid_match = re.search(r'embed/([a-zA-Z0-9_-]+)', src)
            if vid_match:
                video_id = vid_match.group(1)
                data['url_youtube'] = f"https://www.youtube.com/watch?v={video_id}"

        # --- Spotify URL 清理 ---
        # 目標格式: https://open.spotify.com/track/{id}
        if data['id']:
            data['url_spotify'] = f"https://open.spotify.com/track/{data['id']}"
        else:
            data['url_spotify'] = ""

        # --- 流派 (Genres) ---
        spotify_tags = soup.find('div', id='spotify-tags')
        if spotify_tags:
            tags = [a.get_text(strip=True) for a in spotify_tags.find_all('a')]
            data['genres_spotify'] = ", ".join(tags)
        else:
            data['genres_spotify'] = ""

        wiki_content_div = soup.find('div', id='wiki-genres')
        
        # 備用：如果找不太到 id，找外層 class="wiki-tags"
        if not wiki_content_div:
            outer_wiki = soup.find('div', class_='wiki-tags')
            if outer_wiki:
                wiki_content_div = outer_wiki.find('div', id='wiki-genres')

        if wiki_content_div:
            # 取得該 div 內的純文字，並去除空白
            text_content = wiki_content_div.get_text(strip=True)
            
            # 情況 A: 內容是 "---" 或完全空白，代表網站確認無資料
            if text_content == "---" or text_content == "":
                data['genres_wikipedia'] = "N/A"
            
            # 情況 B: 有 <a> 標籤，代表有資料
            else:
                tags = [a.get_text(strip=True) for a in wiki_content_div.find_all('a')]
                if tags:
                    data['genres_wikipedia'] = ", ".join(tags)
                else:
                    # 有 div 但沒 a 標籤，內容也不是 "---"，可能是特殊狀況，保守起見設為 N/A
                    data['genres_wikipedia'] = "N/A"
        else:
            # 情況 C: 根本找不到 <div id="wiki-genres">
            # 這通常代表網頁載入不完全 (AJAX 失敗)，或者是被隱藏
            # 存成空字串，讓 fix_data.py 之後有機會去重抓
            data['genres_wikipedia'] = ""

    # ==========================================
    # 2. 解析數值分析 (維持原樣)
    # ==========================================
    a_section = soup.find('article', id='result-analyzer')
    attributes = ['bpm', 'energy', 'danceability', 'happiness', 'acousticness', 
                  'instrumentalness', 'liveness', 'speechiness', 'loudness', 'popularity', 'duration']
    for attr in attributes:
        if attr not in data: data[attr] = 0
    data['key'] = ""
    data['camelot'] = ""

    if a_section:
        text_content = a_section.get_text()
        
        # 數值解析
        bpm_match = re.search(r'Tempo:\s*(\d+)', text_content)
        if bpm_match: data['bpm'] = int(bpm_match.group(1))

        loud_match = re.search(r'Loudness:\s*(-?\d+)', text_content)
        if loud_match: data['loudness'] = int(loud_match.group(1))

        dur_match = re.search(r'Length:\s*(\d+):(\d+)', text_content)
        if dur_match:
            data['duration'] = int(dur_match.group(1)) * 60 + int(dur_match.group(2))

        key_match = re.search(r'Key:\s*(.+?)\s+Camelot', text_content)
        data['key'] = key_match.group(1).strip() if key_match else ""

        cam_match = re.search(r'Camelot:\s*(\w+)', text_content)
        data['camelot'] = cam_match.group(1).strip() if cam_match else ""

        # 透過進度條百分比抓取特徵
        map_keyword_field = {
            'Popularity': 'popularity',
            'Happiness': 'happiness',
            'Danceability': 'danceability',
            'Energy': 'energy',
            'Acousticness': 'acousticness',
            'Instrumentalness': 'instrumentalness',
            'Liveness': 'liveness',
            'Speechiness': 'speechiness'
        }
        for label, field in map_keyword_field.items():
            pattern = re.compile(rf"{label}:\s*(\d+)/100")
            match = pattern.search(text_content)
            if match:
                data[field] = int(match.group(1))

    data['cluster'] = None 
    return data


def run_scraper():

    options = webdriver.ChromeOptions()
    # options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--mute-audio")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    options.add_argument(f'user-agent={user_agent}')
    # options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        topic_scraper = Chosic_Topics_Scraper(driver)
        topic_urls = topic_scraper.scrape()

        # 從第幾個主題開始爬（0 = 全部；中斷後續爬時可調大）
        target_topics = topic_urls[START_TOPIC_INDEX:]
        
        for topic_url in target_topics:
            list_scraper = Chosic_Songlist_Scraper(driver, topic_url)
            song_ids = list_scraper.scrape()

            # 測試用：只抓前 2 首歌
            target_songs = song_ids

            for song_id in target_songs:
                analyzer = Chosic_Song_Analysis_Scraper(driver, song_id)
                html = analyzer.scrape()
                clean_data = parse_html_data(html, song_id)
                save_song(clean_data)
                
                time.sleep(random.uniform(2, 4))

    finally:
        driver.quit()

if __name__ == "__main__":
    init_db()
    run_scraper()