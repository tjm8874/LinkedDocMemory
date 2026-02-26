import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# OpenAIライブラリのインポート（様々なAPIベンダーに対応可能）
from openai import OpenAI

def get_llm_client():
    """
    .env設定からLLMクライアント（OpenAI互換）を生成して返す
    """
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    
    if not api_key:
         raise ValueError("LLM_API_KEY is not set in .env")
         
    # BASE_URLが指定されていない場合はデフォルト（OpenAI公式等）になる
    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None
    )
    return client

def evaluate_association(doc_a_name: str, doc_a_text: str, doc_b_name: str, doc_b_text: str) -> float:
    """
    2つのドキュメント間の関連性をLLMに評価させ、0.0〜1.0のスコアを返す。
    """
    client = get_llm_client()
    model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini") # デフォルトモデル
    
    prompt = f"""
あなたは非常に優秀なナレッジグラフ・エージェントです。
以下の2つのドキュメントの内容を読み、それらが「意味的に」「文脈的に」どの程度関連しているかを 0.0 から 1.0 の数値で評価してください。

評価基準:
- 1.0 : 全く同じテーマを扱っていたり、一方が他方の直接的な続きや詳細解説になっている。強い関連性がある。
- 0.5 : 一部の概念が共通している、あるいは組み合わせて読むと有益な気づきがある。
- 0.0 : 全く無関係である。

[ドキュメントA: {doc_a_name}]
{doc_a_text[:1500]}... (省略)

[ドキュメントB: {doc_b_name}]
{doc_b_text[:1500]}... (省略)

回答は関連度（数値）のみを出力してください。例: 0.65
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "あなたは関連度を数値で返すアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        # 数値のみを抽出
        result_text = response.choices[0].message.content.strip()
        # 簡易的な安全処理（数字以外の文字が含まれていた場合のパース）
        import re
        match = re.search(r'0\.[0-9]+|1\.0', result_text)
        if match:
            score = float(match.group())
            return min(max(score, 0.0), 1.0)
        return 0.0
        
    except Exception as e:
        print(f"[LLM Error] Failed to evaluate association: {e}")
        return 0.0
