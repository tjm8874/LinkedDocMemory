import os
from dotenv import load_dotenv
from retriever_core import MemoryRetriever

# インデックスをメモリにキャッシュするためのモジュールレベル変数
_retriever_instance = None

def get_retriever():
    """Retrieverインスタンスを初期化し、シングルトンとして返す"""
    global _retriever_instance
    if _retriever_instance is None:
        load_dotenv()
        # .env または デフォルト値
        docs_dir = os.environ.get("DOCS_DIR", "./SampleDocs")
        
        # 相対パスの場合は、このスクリプトの位置を基準に解決する
        if not os.path.isabs(docs_dir):
            base_path = os.path.dirname(os.path.abspath(__file__))
            docs_dir = os.path.join(base_path, docs_dir)
            
        _retriever_instance = MemoryRetriever(docs_dir)
        print(f"[Info] Initialized MemoryRetriever with docs_dir: {docs_dir}")
        print(f"[Info] Loaded {len(_retriever_instance.documents)} documents.")
    return _retriever_instance

def retrieve_context(keyword: str, max_length: int = 4096) -> str:
    """
    [Tool/Skill関数]
    フリーテキストのキーワード(keyword)から、最大文字数(max_length)に収まる範囲で
    関連するコンテキスト(テキストの連結文字列)を抽出して返します。
    """
    try:
        retriever = get_retriever()
        # Spreading Activationを伴う検索の実行 (上位5件取得の例)
        results = retriever.retrieve(keyword, top_k=5)
        
        if not results:
            return "関連するコンテキストが見つかりませんでした。"
            
        context_parts = []
        current_length = 0
        
        for doc_name, score in results:
            text = retriever.documents[doc_name]
            
            # ドキュメント間の区切りとヘッダー
            header = f"\n\n--- Document: {doc_name} (Score: {score:.3f}) ---\n\n"
            part_len = len(header) + len(text)
            
            # 文字数上限チェック
            if current_length + len(header) >= max_length:
                # ヘッダーすら入らないなら終了
                break

            if current_length + part_len > max_length:
                # このドキュメントは途中で切り捨てて終了
                allowed_len = max_length - current_length - len(header)
                if allowed_len > 0:
                    truncated_text = text[:allowed_len] + "\n... (省略されました: コンテキスト上限到達)"
                    context_parts.append(header + truncated_text)
                break
            else:
                context_parts.append(header + text)
                current_length += part_len
                
        final_context = "".join(context_parts).strip()
        return final_context
        
    except Exception as e:
        return f"コンテキスト取得中にエラーが発生しました: {str(e)}"
