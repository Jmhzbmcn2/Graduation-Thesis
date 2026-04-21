import os
import sys
import ast
import pandas as pd
from dotenv import load_dotenv

# Import từ llama-index
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.llms.openai_like import OpenAILike

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    # 1. Load API Key
    load_dotenv()
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Vui lòng thiết lập OPENROUTER_API_KEY trong file .env")
        return

    # 2. Khởi tạo LLM và Evaluator
    print("Đang khởi tạo LlamaIndex FaithfulnessEvaluator qua OpenRouter...")
    llm = OpenAILike(
        model="qwen/qwen3-30b-a3b-instruct-2507", 
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        is_chat_model=True,
        default_headers={
            "HTTP-Referer": "https://github.com/vuduylinh",
            "X-Title": "LightRAG-Eval"
        }
    )
    evaluator = FaithfulnessEvaluator(llm=llm)

    # 3. Đọc dữ liệu từ file Excel
    excel_path = "testcase/eval_ragas_beam.xlsx"
    sheet_name = "Hybrid"
    
    if not os.path.exists(excel_path):
        print(f"Không tìm thấy file: {excel_path}")
        return

    print(f"Đang đọc dữ liệu từ {excel_path}...")
    # Đọc toàn bộ các sheet để lúc lưu lại không bị mất các sheet khác (Hybrid, Mix, Summary...)
    all_sheets = pd.read_excel(excel_path, sheet_name=None)
    
    if sheet_name not in all_sheets:
        print(f"Lỗi: Không tìm thấy sheet '{sheet_name}' trong file.")
        return

    df = all_sheets[sheet_name]
    
    # Tạo các cột mới nếu chưa có
    if "llama_index_faithfulness" not in df.columns:
        df["llama_index_faithfulness"] = None
    if "llama_index_feedback" not in df.columns:
        df["llama_index_feedback"] = None

    # Bạn báo đã chạy RAGAS khoảng 20 câu, ta sẽ chạy lại LlamaIndex trên toàn bộ 20 câu này
    start_index = 0 
    
    print(f"Bắt đầu đánh giá từ dòng {start_index} (index trong DataFrame)...")
    
    for index, row in df.iterrows():
        if index < start_index:
            continue
            
        # Có thể skip nếu đã có dữ liệu ở cột này rồi (uncomment nếu cần)
        # if pd.notna(row["llama_index_faithfulness"]):
        #     continue

        query = str(row["user_input"])
        response = str(row["response"])
        contexts_raw = row["retrieved_contexts"]
        
        # retrieved_contexts thường được lưu ở dạng chuỗi biểu diễn list "['context1', 'context2']" trong excel
        # Nên dùng ast.literal_eval để parse an toàn về kiểu list
        try:
            if isinstance(contexts_raw, str):
                contexts = ast.literal_eval(contexts_raw)
            else:
                contexts = list(contexts_raw)
        except Exception:
            # Nếu lỗi parse (format không đúng), thì cho nguyên chuỗi vào 1 list
            contexts = [str(contexts_raw)]

        print(f"\n[{index}] Đang đánh giá: {query[:60]}...")
        
        try:
            result = evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts
            )
            df.at[index, "llama_index_faithfulness"] = result.score
            df.at[index, "llama_index_feedback"] = result.feedback
            print(f"  -> Điểm: {result.score} | Pass: {result.passing}")
        except Exception as e:
            print(f"  -> [Lỗi]: {e}")
            df.at[index, "llama_index_faithfulness"] = "ERROR"
            df.at[index, "llama_index_feedback"] = str(e)

        # Lưu trung gian sau mỗi 5 câu vào file excel để tránh mất dữ liệu nếu gián đoạn
        if index > 0 and index % 5 == 0:
            all_sheets[sheet_name] = df
            with pd.ExcelWriter("testcase/eval_ragas_beam_out.xlsx") as writer:
                for s_name, s_df in all_sheets.items():
                    s_df.to_excel(writer, sheet_name=s_name, index=False)
            print(f"  [Đã lưu trung gian vào testcase/eval_ragas_beam_out.xlsx]")

    # 4. Lưu lại toàn bộ kết quả vào file ban đầu khi hoàn thành
    all_sheets[sheet_name] = df
    with pd.ExcelWriter("testcase/eval_ragas_beam_out.xlsx") as writer:
        for s_name, s_df in all_sheets.items():
            s_df.to_excel(writer, sheet_name=s_name, index=False)
            
    print(f"\nĐã hoàn thành! Toàn bộ kết quả đã được lưu vào testcase/eval_ragas_beam_out.xlsx")

if __name__ == "__main__":
    main()
