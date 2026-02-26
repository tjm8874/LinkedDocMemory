import os
import re
import networkx as nx
from rank_bm25 import BM25Okapi
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True

class MemoryRetriever:
    def __init__(self, docs_dir):
        """
        ローカルMarkdownフォルダからナレッジグラフとBM25インデックスを構築するクラス
        """
        self.docs_dir = docs_dir
        self.documents = {}      # {ドキュメント名: テキスト本体 (YAML抜き)}
        self.file_paths = {}     # {ドキュメント名: フルパス}
        self.frontmatters = {}   # {ドキュメント名: YAMLディクショナリ}
        self.graph = nx.DiGraph()
        self.bm25 = None
        self.doc_names = []
        
        self.load_documents()
        self.build_graph()
        self.build_bm25_index()

    def parse_markdown(self, text):
        """YAMLフロントマターと純粋なテキストを分離して抽出する"""
        match = re.match(r'^---\n(.*?)\n---\n(.*)', text, flags=re.MULTILINE | re.DOTALL)
        if match:
            yaml_content = match.group(1)
            body = match.group(2)
            try:
                frontmatter = yaml.load(yaml_content) or {}
            except Exception as e:
                print(f"[Warning] Failed to parse YAML: {e}")
                frontmatter = {}
            return frontmatter, body.strip()
        else:
            return {}, text.strip()

    def extract_links(self, text):
        """Wikilink形式 `[[DocumentName]]` または `[[DocumentName|Alias]]` を抽出する"""
        links = re.findall(r'\[\[(.*?)(?:\|.*?)?\]\]', text)
        return [link.strip() for link in links]

    def load_documents(self):
        """指定されたディレクトリからMarkdownファイルをすべて読み込む"""
        if not os.path.exists(self.docs_dir):
            print(f"Directory not found: {self.docs_dir}")
            return
            
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith('.md'):
                    path = os.path.join(root, file)
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        doc_name = os.path.splitext(file)[0]
                        
                        frontmatter, body = self.parse_markdown(content)
                        self.documents[doc_name] = body
                        self.frontmatters[doc_name] = frontmatter
                        self.file_paths[doc_name] = path

    def build_graph(self):
        """Markdown本文のリンクとYAMLの重みから有向グラフ(DiGraph)を構築する"""
        for doc_name, text in self.documents.items():
            self.graph.add_node(doc_name)
            
            # YAMLフロントマターから明示的な外向き(Outbound)リンク強度を取得
            yaml_links = self.frontmatters.get(doc_name, {}).get('links', {})
            
            # 本文から[[]]リンクを抽出
            body_links = self.extract_links(text)
            
            # YAMLに記載があるものはその重みを使用、ないものは本文リンクから均等割り当て計算用
            unweighted_links = [link for link in body_links if link not in yaml_links]
            
            # 手動設定された重みの合計
            yaml_weight_sum = sum(yaml_links.values())
            
            # グラフへの追加
            for link, weight in yaml_links.items():
                if link not in self.graph:
                    self.graph.add_node(link)
                self.graph.add_edge(doc_name, link, weight=float(weight))
                
            # YAMLに記載がない本文リンクへの重みの割り振り (残りの重みを均等割り)
            if unweighted_links:
                remaining_weight = max(0.0, 1.0 - yaml_weight_sum)
                if remaining_weight > 0:
                    weight_per_link = remaining_weight / len(unweighted_links)
                    for link in unweighted_links:
                        if link not in self.graph:
                            self.graph.add_node(link)
                        self.graph.add_edge(doc_name, link, weight=weight_per_link)
        
    def tokenize(self, text):
        """
        日本語・英語ハイブリッド用の簡易バイグラム(N-gram)トークナイザ。
        MeCab等の外部依存をなくすための軽量実装。
        """
        text = text.lower()
        # 空白や改行を削除して連続した文字にする
        text = re.sub(r'\s+', '', text)
        if len(text) < 2:
            return [text]
        return [text[i:i+2] for i in range(len(text)-1)]

    def build_bm25_index(self):
        """読み込んだ全ドキュメントテキストに対してBM25インデックスを構築する"""
        self.doc_names = list(self.documents.keys())
        if not self.doc_names:
            return
            
        tokenized_corpus = [self.tokenize(self.documents[name]) for name in self.doc_names]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, top_k=3, decay_rate=0.8, steps=3, threshold=0.1):
        """
        1. BM25でキーワード検索を行い初期活性値を設定 (Seeding)
        2. NetworkXグラフ上でSpreading Activationによる伝播計算
        3. 最終的な活性値(スコア)が高い順に top_k 件のノードを返す
        """
        if not self.bm25 or not self.doc_names:
            return []
            
        tokenized_query = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 1. 初期活性化 (BM25スコア)
        activations = {name: score for name, score in zip(self.doc_names, bm25_scores) if score > 0}
        
        if not activations:
            return []
            
        # 最大スコアが 1.0 になるように正規化
        max_score = max(activations.values())
        if max_score > 0:
            current_activations = {k: v / max_score for k, v in activations.items()}
        else:
            current_activations = activations.copy()
            
        final_activations = current_activations.copy()
        
        # 2. 活性化拡散 (Spreading Activation)
        for step in range(steps):
            next_activations = {}
            for node, activation in current_activations.items():
                # しきい値以下なら伝播させない事で計算量爆発を抑える
                if activation < threshold:
                    continue
                    
                if node in self.graph:
                    for neighbor in self.graph.neighbors(node):
                        weight = self.graph[node][neighbor].get('weight', 0.0)
                        # リンク強度(重み)と減衰率(Decay)を掛けたエネルギーを隣へ渡す
                        spread = activation * weight * decay_rate
                        next_activations[neighbor] = next_activations.get(neighbor, 0) + spread
                        
            if not next_activations:
                break
                
            # 計算した拡散エネルギーを最終スコアに加算
            for node, act in next_activations.items():
                final_activations[node] = final_activations.get(node, 0) + act
                
            # 次のステップの起点として、エネルギーが1.0を超えすぎないようクリッピング
            current_activations = {k: min(v, 1.0) for k, v in next_activations.items()}
                
        # 3. 最終スコア順に並び替え
        sorted_nodes = sorted(final_activations.items(), key=lambda x: x[1], reverse=True)
        # 実際にテキストが存在するノードのみフィルタ (Danglingリンク先の除外)
        result = [(node, score) for node, score in sorted_nodes if node in self.documents]
        return result[:top_k]
