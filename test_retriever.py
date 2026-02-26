import os
import sys

# テスト用に環境変数を強制設定
os.environ["DOCS_DIR"] = "./SampleDocs"

from memory_tool import retrieve_context, get_retriever

def main():
    print("========== 検索テスト開始 ==========")
    r = get_retriever()
    print("Graph nodes:", r.graph.nodes)
    print("Graph edges:", r.graph.edges(data=True))
    print("\n--------------------------------")
    
    query = "ベクトル"
    print(f"テスト1: キーワード '{query}' の検索 (上限2000文字)")
    result1 = retrieve_context(query, max_length=2000)
    print("【結果】")
    print(result1)
    print("\n--------------------------------")
    
    print(f"テスト2: キーワード '{query}' の検索 (上限500文字制限のテスト)")
    result2 = retrieve_context(query, max_length=500)
    print("【結果】")
    print(result2)
    print("\n--------------------------------")

    # コールドスタートテスト
    query3 = "新規作成 最新メモ"
    print(f"テスト3: コールドスタートテスト - リンクがない新規メモ '{query3}' の検索")
    result3 = retrieve_context(query3, max_length=1000)
    print("【結果】")
    print(result3)
    print("\n--------------------------------")

if __name__ == "__main__":
    main()
