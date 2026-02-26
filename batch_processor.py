import os
import networkx as nx
from ruamel.yaml import YAML

from retriever_core import MemoryRetriever
from llm_association import evaluate_association

yaml = YAML()
yaml.preserve_quotes = True

def run_batch_process():
    print("========== 夜間バッチ処理開始 ==========")
    load_dotenv_or_default()
    docs_dir = os.environ.get("DOCS_DIR", "./SampleDocs")
    
    # 最新のドキュメント状態を読み込み
    retriever = MemoryRetriever(docs_dir)
    print(f"対象ドキュメント数: {len(retriever.doc_names)}")
    
    # === 1. 連想・新規リンク構築 (Association) ===
    print("\n[Phase 1] 連想・新規リンク構築")
    new_links_added = 0
    
    for doc_name in retriever.doc_names:
        # [判断基準]: アウトバウンドエッジ数（発リンク）が0の孤立ドキュメントを探す
        out_degree = retriever.graph.out_degree(doc_name)
        if out_degree == 0:
            print(f"  孤立ドキュメント発見: {doc_name} - LLMによる関連性評価を開始します...")
            doc_text = retriever.documents[doc_name]
            
            # 他のすべてのドキュメントと関連性を評価（全探索は重いため、本来はBM25等で候補を絞るべきだが今回はプロトタイプとして全探索）
            for other_doc in retriever.doc_names:
                if doc_name == other_doc:
                    continue
                
                other_text = retriever.documents[other_doc]
                score = evaluate_association(doc_name, doc_text, other_doc, other_text)
                
                if score >= 0.5: # 関連性の閾値
                    print(f"    -> 関連性を発見: [{other_doc}] (スコア: {score})")
                    add_link_to_file(retriever, doc_name, other_doc, score)
                    new_links_added += 1

    # 最新状態を再読み込み (ファイルが変更されたため)
    if new_links_added > 0:
        print("[Info] リンクが追加されたため、インデックスを再構築します。")
        retriever = MemoryRetriever(docs_dir)
        
    # === 2. 忘却・プルーニング (Forgetting) ===
    print("\n[Phase 2] 忘却処理")
    FORGETTING_THRESHOLD = 0.05
    for doc_name, frontmatter in retriever.frontmatters.items():
        if 'links' in frontmatter:
            links = frontmatter['links']
            keys_to_delete = []
            for link_target, weight in links.items():
                if float(weight) < FORGETTING_THRESHOLD:
                    print(f"  忘却: {doc_name} -> {link_target} (重み: {weight} が閾値未満のため削除)")
                    keys_to_delete.append(link_target)
            
            if keys_to_delete:
                for k in keys_to_delete:
                    del links[k]
                save_frontmatter_to_file(retriever, doc_name, frontmatter)

    # === 3. 正規化 (Consolidation) ===
    print("\n[Phase 3] 正規化処理")
    retriever = MemoryRetriever(docs_dir) # 最新状態再読み込み
    for doc_name, frontmatter in retriever.frontmatters.items():
        if 'links' in frontmatter and frontmatter['links']:
            links = frontmatter['links']
            total_weight = sum(float(w) for w in links.values())
            
            if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
                print(f"  正規化: {doc_name} (現在の合計: {total_weight:.3f} -> 1.0 に再計算)")
                for target in links:
                    links[target] = round(float(links[target]) / total_weight, 3)
                save_frontmatter_to_file(retriever, doc_name, frontmatter)

    print("\n========== 夜間バッチ処理終了 ==========")
    
def add_link_to_file(retriever: MemoryRetriever, source_doc: str, target_doc: str, score: float):
    """
    ファイルにYAMLリンクメタデータと、本文末尾に可視リンクを追記する
    """
    path = retriever.file_paths[source_doc]
    frontmatter = retriever.frontmatters.get(source_doc, {})
    
    if 'links' not in frontmatter:
        # ruamel.yamlでシリアライズ可能な辞書を作成
        from ruamel.yaml.comments import CommentedMap
        frontmatter['links'] = CommentedMap()
    
    frontmatter['links'][target_doc] = score
    
    body = retriever.documents[source_doc]
    
    # 既にリンクが本文にあるかチェック
    if f"[[{target_doc}]]" not in body:
        # 可視リンクの追記
        body += f"\n\n* AI連想リンク: [[{target_doc}]]"
        
    save_full_document(path, frontmatter, body)

def save_frontmatter_to_file(retriever: MemoryRetriever, doc_name: str, new_frontmatter: dict):
    path = retriever.file_paths[doc_name]
    body = retriever.documents[doc_name]
    save_full_document(path, new_frontmatter, body)
    
def save_full_document(path: str, frontmatter: dict, body: str):
    import io
    yaml_str = ""
    if frontmatter:
        buf = io.StringIO()
        yaml.dump(frontmatter, buf)
        yaml_str = f"---\n{buf.getvalue()}---\n"
        
    full_content = yaml_str + body
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(full_content)

def load_dotenv_or_default():
    from dotenv import load_dotenv
    load_dotenv()
    if "DOCS_DIR" not in os.environ:
         os.environ["DOCS_DIR"] = "./SampleDocs"

if __name__ == "__main__":
    run_batch_process()
