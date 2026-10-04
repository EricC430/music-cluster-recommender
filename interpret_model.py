import sqlite3
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from pathlib import Path
import re

DB_PATH = 'music.db'
MODEL_FOLDER = 'models'

# 確保資料夾存在
Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

class ClusterInterpreter:
    def __init__(self):
        self.clf = None
        self.feature_names = None
        self.X = None
        self.y = None

    def load_data(self):
        conn = sqlite3.connect(DB_PATH)
        # 讀取特徵與 K-means 分出來的 label (cluster)
        query = """
        SELECT bpm, energy, danceability, happiness, 
               acousticness, instrumentalness, liveness, speechiness, loudness,
               cluster
        FROM songs
        WHERE cluster IS NOT NULL
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        self.X = df.drop('cluster', axis=1)
        self.y = df['cluster']
        self.feature_names = self.X.columns.tolist()
        
        print(f"📊 Loaded {len(df)} songs for interpretation.")
        return df

    def train_surrogate_tree(self, max_depth=3):
        """
        訓練代理模型。
        【HW8 修正 1】: 將 criterion 改為 'entropy'，符合作業中計算亂度的邏輯。
        """
        print(f"🌲 Training Decision Tree (criterion='entropy', Depth={max_depth})...")
        
        # 使用 entropy 作為分裂標準
        self.clf = DecisionTreeClassifier(
            criterion='entropy',  # 對應 HW8 的 cal_entropy
            max_depth=max_depth, 
            random_state=42
        )
        self.clf.fit(self.X, self.y)

    def print_rules(self):
        """以文字方式印出分群規則"""
        if not self.clf: return
        print("\n📜 Cluster Rules (Text Format):")
        rules = export_text(self.clf, feature_names=self.feature_names)
        print(rules)

    def show_feature_importance(self):
        """
        【HW8 補充】: 顯示特徵重要性 (Feature Importance)。
        這在數學上等同於 HW8 中計算 Information Gain 後的排序結果。
        """
        if not self.clf: return
        print("\nBg Feature Importance (Information Gain):")
        importances = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print(importances)

    def plot_tree_matplotlib(self):
        """
        【HW8 修正 2】: 使用 plot_tree 直接繪圖，不依賴 Graphviz。
        """
        if not self.clf: return
        print("\n🎨 Plotting tree with Matplotlib...")
        
        plt.figure(figsize=(20, 10))  # 設定圖片大小
        annotations = plot_tree(
            self.clf, 
            feature_names=self.feature_names, 
            class_names=[f"C{i}" for i in self.clf.classes_], # 顯示 C0, C1...
            filled=True, 
            rounded=True, 
            fontsize=10,
            impurity=False, # 建議：通常 entropy 也可以隱藏，讓圖更乾淨
            proportion=False
        )
        for text in annotations:
            original_text = text.get_text()
            # 將文字依換行符號切開，過濾掉開頭是 'value' 的行
            new_text = re.sub(r"value = \[.*?\]\n", "", original_text, flags=re.DOTALL)
            text.set_text(new_text)
        plt.title("Cluster Decision Tree (Entropy Based)")
        
        # 存成圖片
        save_path = f"{MODEL_FOLDER}/cluster_tree_plot.png"
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"💾 Plot saved to {save_path}")
        # plt.show() # 如果在 Jupyter notebook 中執行可以打開這行

    def export_graphviz_dot(self):
        """保留原本的輸出 .dot 檔案功能 (作為備用)"""
        path = f"{MODEL_FOLDER}/cluster_tree.dot"
        export_graphviz(
            self.clf,
            out_file=path,
            feature_names=self.feature_names,
            class_names=[f"Cluster {i}" for i in sorted(self.y.unique())],
            filled=True,
            rounded=True
        )
        # print(f"💾 Dot file saved to {path}")

if __name__ == "__main__":
    interpreter = ClusterInterpreter()
    interpreter.load_data()
    
    # 訓練模型 (使用 entropy)
    interpreter.train_surrogate_tree(max_depth=4) 
    
    # 1. 印出規則 (文字版)
    interpreter.print_rules()
    
    # 2. 印出特徵重要性 (對應 HW8 的 Information Gain 排序)
    interpreter.show_feature_importance()
    
    # 3. 繪圖 (Matplotlib 版，對應 HW8 的 tree.plot_tree)
    interpreter.plot_tree_matplotlib()
    
    # 4. 繪圖 (Dot 版，保留備用)
    interpreter.export_graphviz_dot()