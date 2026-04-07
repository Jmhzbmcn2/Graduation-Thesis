import json
import requests
import sys
import os

# Đảm bảo có thể import module lightrag 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lightrag.prompt import PROMPTS

# URL và Headers cho Cloudflare Worker
URL = "https://text-generation.linhngocut1508.workers.dev/"
HEADERS = {
    "Authorization": "Bearer 12345678",
    "Content-Type": "application/json",
}

def main():
    # 1. Các cấu hình cơ bản cho prompt
    tuple_delimiter = PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "<|#|>")
    completion_delimiter = PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "<|COMPLETE|>")
    
    # Sử dụng schema 10 type entities y tế mà bạn đang dùng cho đồ án
    entity_types_list = [
        "Disease", "Symptom", "Drug", "Ingredient", "SideEffect", 
        "DosageForm", "Manufacturer", "TargetGroup", "BodyPart", "Mechanism", "Other"
    ]
    entity_types_str = ", ".join(entity_types_list)
    entity_types_json = json.dumps(entity_types_list)
    language = "Vietnamese"

    # Lấy các ví dụ (nếu có từ PROMPTS)
    examples = "\n".join(PROMPTS.get("entity_extraction_examples", []))

    # 2. Xây dựng System Prompt
    system_prompt = PROMPTS["entity_extraction_system_prompt"].format(
        entity_types=entity_types_str,
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        language=language,
        examples=examples
    )

    # 3. Lấy 1 chunk từ file C:\Users\VUDUYLINH\PycharmProjects\KLTN\LightRAG\data\a-t-ambroxol-an-thien.txt
    text_chunk = """A.T Ambroxol An Thiên là thuốc gì?
Thuốc A.T Ambroxol An Thiên được sản xuất bởi công ty Cổ phần Dược phẩm An Thiên. Công dụng chính của thuốc là làm giảm đờm, nhầy và điều trị các bệnh về đường hô hấp. Thuốc được bào chế dạng dung dịch với hương cam tạo mùi vị dễ chịu hơn khi uống. A.T Ambroxol An Thiên có hai dạng đóng gói: Hộp 30 ống x 5 ml hoặc chai với các thể tích: 30 ml, 60 ml, 100 ml."""

    # 4. Xây dựng User Prompt
    user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
        completion_delimiter=completion_delimiter,
        language=language,
        entity_types=entity_types_json,
        input_text=text_chunk
    )

    print("===== System Prompt =====")
    print(system_prompt[:500] + "...\n(Đã cắt bớt để dễ nhìn)\n")
    
    print("===== User Prompt =====")
    print(user_prompt)
    print("=========================\n")

    # 5. Gửi request đến Cloudflare Worker
    payload = {
        "prompt": user_prompt,
        "systemPrompt": system_prompt,
        "history": []
    }

    try:
        print("Đang gửi request đến Cloudflare Worker...")
        response = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        
        # Kết quả text sinh ra
        result_text = response.text
        print("\n===== KẾT QUẢ TỪ API =====")
        print(result_text)
        print("==========================")
        
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gọi API: {e}")

if __name__ == "__main__":
    main()
