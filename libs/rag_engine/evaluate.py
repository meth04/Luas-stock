import json
import os
import asyncio
import gc
import torch
import pandas as pd
import multiprocessing
import re
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_correctness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from .config import settings
from .retrieval import query_func

def clean_to_atomic_answer(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'(triệu VND|triệu VNĐ|triệu vnd|triệu vnđ|tr\.đ|tr\.đồng)', 'triệu đồng', text, flags=re.IGNORECASE)
    prefixes = [r".* là ", r".* đạt ", r".* ghi nhận "]
    for p in prefixes:
        text = re.sub(p, "", text)
    return text.strip().replace("(", "").replace(")", "")

def strict_normalize(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'(triệu VND|triệu VNĐ|triệu vnd|triệu vnđ|tr\.đ|tr\.đồng)', 'triệu đồng', text, flags=re.IGNORECASE)
    text = re.sub(r'^(.* là |.* đạt |.* ghi nhận là )', '', text, flags=re.IGNORECASE)
    return text.strip().replace("(", "").replace(")", "")


def run_isolated_query(question):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        raw, refined = loop.run_until_complete(query_func(None, question, "hybrid"))
        return {"raw_contexts": raw, "refined": refined}
    except Exception as e:
        print(e)
        return {"raw_contexts": str(e), "refined": "Lỗi hệ thống."}
    finally:
        loop.close()

def truncate_context(text: str, max_chars: int = 20000):
    return text[:max_chars] + "... [TRUNCATED]" if len(text) > max_chars else text

NUM_RE = re.compile(r"[-+]?\d[\d\.,]*")

def pick_supporting_contexts(response: str, contexts: list[str], keep_fallback: int = 5):
    nums = NUM_RE.findall(response or "")
    nums = list(dict.fromkeys(nums)) 
    if not nums:
        return contexts[:keep_fallback]

    supporting = []
    for c in contexts:
        if any(n in c for n in nums):
            supporting.append(c)

    return supporting[:keep_fallback] if supporting else contexts[:keep_fallback]


async def run_eval_async(data_dir: str):
    dataset_path = os.path.join(data_dir, "golden_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"LỖI: Không tìm thấy file {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # --- GIAI ĐOẠN 1: SINH CÂU TRẢ LỜI ---
    print(f"\n{'='*60}\nGIAI ĐOẠN 1: SINH CÂU TRẢ LỜI\n{'='*60}")
    
    qa_results = []
    for i, item in enumerate(golden_data):
        q = item["query"]
        print(f"[{i+1}/{len(golden_data)}] ĐANG XỬ LÝ: {q[:100]}")
        
        # Windows multiprocessing safe: sử dụng spawn
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            result = pool.apply(run_isolated_query, (q,))
        
        print(f"👉 ĐÁP ÁN: {result['refined']}")
        
        contexts = result.get("raw_contexts") or []
        contexts = [truncate_context(c) for c in contexts if isinstance(c, str) and len(c.strip()) > 0]

        contexts_for_faith = pick_supporting_contexts(result["refined"], contexts, keep_fallback=5)

        qa_results.append({
            "user_input": q,
            "reference": item["ground_truth_answer"],
            "response": result["refined"],
            "retrieved_contexts": contexts_for_faith,
            "idx": i
        })
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- GIAI ĐOẠN 2: ĐÁNH GIÁ RAGAS ---
    print(f"\n{'='*60}\nGIAI ĐOẠN 2: ĐÁNH GIÁ \n{'='*60}")
    
    os.environ["OPENAI_API_KEY"] = settings.API_KEY
    os.environ["OPENAI_API_BASE"] = settings.BASE_URL
    
    financial_judge_instruction = (
        "ROLE: Bạn là Chuyên gia Kiểm toán Cấp cao (Senior Auditor) kiêm Giám khảo AI Khắt khe (Strict Judge). "
        "NHIỆM VỤ: Đánh giá hệ thống RAG dựa trên các bằng chứng được cung cấp. "
        
        "--- PHẦN 1: QUY TẮC CHẤM ĐIỂM FAITHFULNESS (ĐỘ TRUNG THỰC) - CỰC KỲ KHẮT KHE --- "
        "1. NGUYÊN TẮC 'ZERO TOLERANCE': "
        "   - Nếu 'Câu trả lời' (Response) chứa bất kỳ con số nào KHÔNG XUẤT HIỆN trong 'Ngữ cảnh' (Context) -> Đánh giá là BỊA ĐẶT (Hallucination). "
        "   - Ví dụ: Context có '10 tỷ', Response ghi '10.000 triệu' (tự quy đổi mà context không nói) -> Có thể bị coi là không trung thực nếu context không chứa thông tin quy đổi. "
        "2. KHÔNG DÙNG KIẾN THỨC NGOÀI: "
        "   - AI chỉ được trả lời dựa trên Context. Nếu Context sai (ví dụ OCR lỗi: 1+1=5), AI trung thực phải trả lời là 5. Nếu AI trả lời là 2 -> Faithfulness = 0. "
        "3. PHẠT NẶNG SUY DIỄN: "
        "   - Nếu AI tự ý cộng trừ nhân chia phức tạp mà không có trong văn bản gốc -> Trừ điểm Faithfulness. "

        "--- PHẦN 2: QUY TẮC CHẤM ĐIỂM CORRECTNESS (ĐỘ CHÍNH XÁC) - LINH HOẠT TOÁN HỌC --- "
        "1. ƯU TIÊN GIÁ TRỊ TUYỆT ĐỐI: "
        "   - Khi so sánh Response với Ground Truth (Reference), hãy bỏ qua định dạng, chỉ so sánh GIÁ TRỊ SỐ HỌC. "
        "   - Quy tắc: '(Số)' == '-Số' (Số âm). "
        "   - Quy tắc: '1 tỷ' == '1.000 triệu' == '1.000.000.000'. "
        "2. CHẤP NHẬN LÀM TRÒN: "
        "   - Chênh lệch nhỏ ở hàng đơn vị do làm tròn (ví dụ: .01) được chấp nhận là ĐÚNG (Correct). "

        "TÓM LẠI: "
        "- Khi chấm Faithfulness: Hãy soi mói như một Cảnh sát (Chỉ tin vào văn bản Context). "
        "- Khi chấm Correctness: Hãy tư duy như một Kế toán trưởng (Chỉ quan tâm giá trị cuối cùng)."
    )
    llm = ChatOpenAI(
        model=settings.JUDGE_MODEL, 
        temperature=0,
        model_kwargs={"extra_body": {"system_prompt": financial_judge_instruction}}
    )
    emb = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL, model_kwargs={'device': 'cuda'})
    run_config = RunConfig(timeout=300, max_retries=3, max_workers=12)

    # --- PASS 1: RAW DATA ---
    print("Pass 1: Đánh giá Faithfulness, Relevancy, Precision, Recall...")
    ds_raw = Dataset.from_dict({
        "user_input": [x["user_input"] for x in qa_results],
        "response": [x["response"] for x in qa_results],
        "retrieved_contexts": [x["retrieved_contexts"] for x in qa_results],
        "reference": [x["reference"] for x in qa_results]
    })
    
    res_raw_obj = evaluate(
        dataset=ds_raw,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm, embeddings=emb, run_config=run_config
    )

    # --- PASS 2: NORMALIZED DATA ---
    print("Pass 2: Đánh giá Answer Correctness")
    ds_norm = Dataset.from_dict({
        "user_input": [x["user_input"] for x in qa_results],
        "response": [strict_normalize(x["response"]) for x in qa_results],
        "retrieved_contexts": [x["retrieved_contexts"] for x in qa_results],
        "reference": [strict_normalize(x["reference"]) for x in qa_results]
    })
    
    res_norm_obj = evaluate(
        dataset=ds_norm,
        metrics=[answer_correctness],
        llm=llm, embeddings=emb, run_config=run_config
    )

    # --- TỔNG HỢP KẾT QUẢ  ---
    raw_avg = res_raw_obj.to_pandas().mean(numeric_only=True).to_dict()
    norm_avg = res_norm_obj.to_pandas().mean(numeric_only=True).to_dict()
    
    final_scores = {**raw_avg, **norm_avg}

    print("\n" + "*"*60)
    print("KẾT QUẢ RAGAS CUỐI CÙNG (DỮ LIỆU TÀI CHÍNH):")
    for metric, score in final_scores.items():
        print(f" - {metric}: {score:.4f}")
    print("*"*60)
    
    # Gộp Dataframe để xuất CSV chi tiết
    df_raw = res_raw_obj.to_pandas()
    df_norm = res_norm_obj.to_pandas()
    
    # Kết hợp các cột điểm và metadata
    df_raw["answer_correctness"] = df_norm["answer_correctness"]
    df_raw["normalized_response"] = ds_norm["response"]
    df_raw["normalized_reference"] = ds_norm["reference"]
    
    output_file = "ragas_report_optimized.csv"
    df_raw.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Báo cáo chi tiết đã lưu vào '{output_file}'")

def run_eval(data_dir: str):
    multiprocessing.freeze_support()
    asyncio.run(run_eval_async(data_dir))