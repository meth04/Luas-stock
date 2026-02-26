import os
import asyncio
import re
import sys
from lightrag import QueryParam
from .core import get_rag_engine
from .llm import openai_complete_if_cache
from .config import settings
from datetime import datetime

# Danh sách 30 mã VN30 chính xác
VN30_MAPPING = {
    "ACB": ["ACB", "Á CHÂU"], "BCM": ["BCM", "BECAMEX"], "BID": ["BID", "BIDV", "ĐẦU TƯ VÀ PHÁT TRIỂN"],
    "CTG": ["CTG", "VIETINBANK", "CÔNG THƯƠNG"], "DGC": ["DGC", "ĐỨC GIANG"], "FPT": ["FPT"],
    "GAS": ["GAS", "PV GAS"], "GVR": ["GVR", "CAO SU"], "HDB": ["HDB", "HDBANK"],
    "HPG": ["HPG", "HÒA PHÁT"], "LPB": ["LPB", "LPBANK", "LỘC PHÁT", 'LP Bank'], "MBB": ["MBB", "MB BANK", "QUÂN ĐỘI"],
    "MSN": ["MSN", "MASAN"], "MWG": ["MWG", "THẾ GIỚI DI ĐỘNG"], "PLX": ["PLX", "PETROLIMEX"],
    "SAB": ["SAB", "SABECO"], "SHB": ["SHB"], "SSB": ["SSB", "SEABANK"], "SSI": ["SSI"],
    "STB": ["STB", "SACOMBANK"], "TCB": ["TCB", "TECHCOMBANK"], "TPB": ["TPB", "TPBANK", "TIÊN PHONG"],
    "VCB": ["VCB", "VIETCOMBANK", "NGOẠI THƯƠNG"], "VHM": ["VHM", "VINHOMES"], "VIB": ["VIB"],
    "VIC": ["VIC", "VINGROUP"], "VJC": ["VJC", "VIETJET"], "VNM": ["VNM", "VINAMILK"],
    "VPB": ["VPB", "VPBANK"], "VRE": ["VRE", "VINCOM RETAIL"]
}

FINANCIAL_CODE_MAPPING = {
    # Bảng Cân đối kế toán
    "TỔNG TÀI SẢN": ["Mã số 270", "Tổng tài sản"],
    "TIỀN": ["Mã số 110", "Tiền và các khoản tương đương tiền"],
    "ĐẦU TƯ TÀI CHÍNH": ["Mã số 120", "Mã số 250"],
    "PHẢI THU": ["Mã số 130", "Mã số 210"],
    "HÀNG TỒN KHO": ["Mã số 140", "Hàng tồn kho"],
    "TÀI SẢN CỐ ĐỊNH": ["Mã số 220"],
    "NỢ PHẢI TRẢ": ["Mã số 300", "Tổng nợ phải trả"],
    "VAY": ["Mã số 320", "Mã số 338", "Vay và nợ thuê tài chính"],
    "VỐN CHỦ SỞ HỮU": ["Mã số 400", "Mã số 410"],
    
    # Kết quả kinh doanh (Doanh nghiệp)
    "DOANH THU": ["Mã số 01", "Mã số 10", "Doanh thu thuần"],
    "GIÁ VỐN": ["Mã số 11", "Giá vốn hàng bán"],
    "LỢI NHUẬN GỘP": ["Mã số 20"],
    "DOANH THU TÀI CHÍNH": ["Mã số 21"],
    "CHI PHÍ TÀI CHÍNH": ["Mã số 22"],
    "CHI PHÍ BÁN HÀNG": ["Mã số 25"],
    "CHI PHÍ QUẢN LÝ": ["Mã số 26"],
    "LỢI NHUẬN THUẦN": ["Mã số 30"],
    "LỢI NHUẬN TRƯỚC THUẾ": ["Mã số 50", "Tổng lợi nhuận kế toán trước thuế"],
    "LỢI NHUẬN SAU THUẾ": ["Mã số 60", "Lợi nhuận sau thuế thu nhập doanh nghiệp"],
    
    # Kết quả kinh doanh (Ngân hàng - Đặc thù)
    "THU NHẬP LÃI THUẦN": ["Thu nhập lãi thuần", "I. Thu nhập lãi thuần"],
    "LÃI THUẦN DỊCH VỤ": ["Lãi thuần từ hoạt động dịch vụ"],
    "DỰ PHÒNG RỦI RO": ["Chi phí dự phòng rủi ro tín dụng"],
    
    # Lưu chuyển tiền tệ
    "LƯU CHUYỂN TIỀN TỪ HĐKD": [
    "Mã số 20", 
    "Lưu chuyển tiền thuần từ hoạt động kinh doanh", 
    "Dòng tiền thuần từ hoạt động kinh doanh"
    ],
    "LƯU CHUYỂN TIỀN TỪ ĐẦU TƯ": ["Mã số 30", "Lưu chuyển tiền thuần từ hoạt động đầu tư"],
    "LƯU CHUYỂN TIỀN TỪ TÀI CHÍNH": ["Mã số 40", "Lưu chuyển tiền thuần từ hoạt động tài chính"],

}

def get_financial_hints_and_report_type(question):
    """
    Trả về: (Danh sách mã số gợi ý, Loại báo cáo cần tìm)
    """
    q_upper = question.upper()
    hints = []
    report_type = ""

    # Xác định loại báo cáo để khoanh vùng tìm kiếm
    if any(x in q_upper for x in ["TÀI SẢN", "NỢ", "VỐN", "VAY", "TIỀN GỬI"]):
        report_type = "Bảng cân đối kế toán"
    elif any(x in q_upper for x in ["LỢI NHUẬN", "DOANH THU", "LÃI", "CHI PHÍ"]):
        report_type = "Báo cáo kết quả hoạt động kinh doanh"
    elif any(x in q_upper for x in ["LƯU CHUYỂN", "DÒNG TIỀN", "TIỀN THU", "TIỀN CHI"]):
        report_type = "Báo cáo lưu chuyển tiền tệ"

    # Lấy mã số
    for key, codes in FINANCIAL_CODE_MAPPING.items():
        # Check kỹ hơn: key phải nằm trọn vẹn trong câu hỏi hoặc các từ khóa chính khớp
        if key in q_upper:
            hints.extend(codes)
        # Fallback cho trường hợp viết tắt: "LCTT" -> Lưu chuyển tiền tệ
        if "LCTT" in q_upper and "LƯU CHUYỂN" in key:
            hints.extend(codes)

    return list(set(hints))[:4], report_type


def identify_ticker_python_fallback(question):
    q_upper = question.upper()
    for ticker, keywords in VN30_MAPPING.items():
        if any(kw in q_upper for kw in keywords):
            return ticker
    return None

def remove_think_tag(text):
    """Loại bỏ nội dung trong thẻ <think> để lấy đáp án cuối cùng"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

async def extract_metadata_smart(question):
    """
    Kết hợp Python (nhận diện Tên công ty) + LLM (suy luận Thời gian).
    """
    now = datetime.now()
    current_date_str = now.strftime("%d/%m/%Y")
    
    detected_ticker = identify_ticker_python_fallback(question)
    
    ticker_hint = f"Đã phát hiện mã: {detected_ticker}" if detected_ticker else "Chưa xác định được mã, hãy tự suy luận từ tên công ty."

    mapping_str = "\n".join([f"- {k}: {', '.join(v)}" for k, v in VN30_MAPPING.items()])

    prompt = f"""
    ROLE: Bạn là trợ lý Routing dữ liệu tài chính.
    THỜI GIAN HIỆN TẠI: {current_date_str}
    
    DANH SÁCH MAPPING (MÃ: Tên gọi phổ biến):
    {mapping_str}
    
    INPUT QUESTION: "{question}"
    HINT: {ticker_hint}

    QUY TẮC SUY LUẬN:
    TICKER: Ưu tiên lấy theo HINT. Nếu HINT chưa có, hãy tra từ điển Mapping trên để tìm mã phù hợp nhất.
    QUY TẮC SUY LUẬN THỜI GIAN (Logic Financial Reporting):
        1. Nếu câu hỏi không nhắc đến thời gian (Ví dụ: "Lợi nhuận VCB bao nhiêu?"):
        - Mặc định là báo cáo quý GẦN NHẤT đã hoàn thành so với hiện tại.
        - Ví dụ: Hôm nay 13/02/2026 -> Báo cáo mới nhất là Q4 2025 (vì Q1 2026 chưa hết).
        2. Nếu câu hỏi dùng từ tương đối ("năm ngoái", "quý trước"):
        - Tính toán dựa trên THỜI GIAN HIỆN TẠI.
        3. Nếu câu hỏi chỉ có năm mà không có quý (Ví dụ: "Năm 2025"):
        - Mặc định trả về "Q4" (thường là báo cáo tổng kết năm).
        4. Nếu câu hỏi có thời gian:
        - Ví dụ như trong câu hỏi chứa các từ khóa như 'tháng 9 năm 2025' hoặc '30 tháng 9 năm 2025' thì lấy quý 3 năm 2025
        - Ví dụ như thời gian '31 tháng 12 năm 2025' hoặc tương tự thì lấy quý 4 năm 2025
        4. Xác định Mã chứng khoán (Ticker) từ tên công ty hoặc mã (Ví dụ: "Thế giới di động" -> MWG).

    OUTPUT FORMAT (Chỉ 1 dòng duy nhất): TICKER|YEAR|QUARTER
    Ví dụ: SSB|2025|Q3
    """

    try:
        res = await openai_complete_if_cache(
            prompt, 
            model_name=settings.LLM_MODEL, 
            temperature=0.0,
            max_tokens=50
        )
        
        parts = res.strip().split('|')
        if len(parts) >= 3:
            ticker = parts[0].strip().upper()
            year = parts[1].strip()
            quarter = parts[2].strip().upper()
            
            if ticker not in VN30_MAPPING:
                fallback = identify_ticker_python_fallback(question)
                if fallback: ticker = fallback
                else: ticker = "DEFAULT"
                
            return ticker, year, quarter
            
    except Exception as e:
        print(f"[ROUTING ERROR] {e}", file=sys.stderr)

    # Fallback cuối cùng
    return "DEFAULT", "2025", "Q3"


def identify_ticker_python(question):
    q = question.upper()
    for ticker, keywords in VN30_MAPPING.items():
        if any(kw in q for kw in keywords):
            return ticker
    return "DEFAULT"

async def generate_search_queries(question: str, ticker: str, year: str, quarter: str):
    """
    Chiến lược sinh Query thế hệ 3: Multi-Industry & Deep Anchor.
    Bao phủ: Sản xuất, Ngân hàng, Bất động sản, Chứng khoán.
    """
    time_period = f"{quarter}/{year}"
    
    # Prompt được thiết kế như một chuyên gia Kế toán trưởng (Chief Accountant)
    prompt = f"""
    ROLE: Bạn là Chuyên gia Phân tích Dữ liệu Tài chính Việt Nam (VAS Expert).
    NHIỆM VỤ: Sinh ra 3-4 truy vấn tìm kiếm (Search Queries) tối ưu nhất để tìm số liệu trong tài liệu OCR.
    
    INPUT:
    - Câu hỏi: "{question}"
    - Mã CK: {ticker} | Kỳ báo cáo: {time_period}
    
    KIẾN THỨC NGÀNH (BẮT BUỘC ÁP DỤNG):
    1. **NGÀNH NGÂN HÀNG (VD: VCB, BID, CTG, TCB, MBB, ACB...)**:
       - Không dùng "Doanh thu bán hàng", hãy tìm "Thu nhập lãi thuần" hoặc "Thu nhập lãi và các khoản thu nhập tương tự".
       - Không dùng "Giá vốn", hãy tìm "Chi phí lãi".
       - "Nợ vay" thường là "Tiền gửi của khách hàng" hoặc "Phát hành giấy tờ có giá".
       - "Dự phòng" là "Chi phí dự phòng rủi ro tín dụng".
       
    2. **DOANH NGHIỆP THƯỜNG (VD: HPG, VNM, MSN, MWG...)**:
       - Dùng "Doanh thu thuần", "Giá vốn hàng bán", "Lợi nhuận gộp".
       - Bảng CĐKT: Tìm "Tổng tài sản" (Mã 270), "Hàng tồn kho" (Mã 140), "Nợ phải trả" (Mã 300).
       
    3. **BẤT ĐỘNG SẢN (VD: VHM, NVL, BCM...)**:
       - Chú ý "Người mua trả tiền trước" (Khoản mục trọng yếu).
       - "Chi phí xây dựng cơ bản dở dang" (Mã 242).

    CHIẾN THUẬT "MỎ NEO" (ANCHORING) - ĐỂ TÌM ĐÚNG BẢNG:
    - Nếu hỏi **Lưu chuyển tiền tệ**:
      + Query 1: "{ticker} Báo cáo lưu chuyển tiền tệ {year}"
      + Query 2 (Gián tiếp): "{ticker} {year} Khấu hao tài sản cố định" (Từ khóa luôn có trong LCTT gián tiếp)
      + Query 3 (Trực tiếp - Ngân hàng): "{ticker} {year} Tiền chi trả cho nhân viên"
      + Query 4 (Mã số): "{ticker} {year} Mã số 20" hoặc "{ticker} {year} Mã số 30"
      
    - Nếu hỏi **Kết quả kinh doanh**:
      + Query 1: "{ticker} Báo cáo kết quả hoạt động kinh doanh {year}"
      + Query 2: "{ticker} {year} Lợi nhuận sau thuế thu nhập doanh nghiệp" (Full phrase)
      + Query 3: "{ticker} {year} Tổng lợi nhuận kế toán trước thuế"
      
    - Nếu hỏi **Thuyết minh/Chi tiết** (VD: Cơ cấu nợ, Chi tiết hàng tồn kho):
      + Query 1: "{ticker} Thuyết minh báo cáo tài chính {year}"
      + Query 2: "Thuyết minh {question} {year}"

    YÊU CẦU ĐẦU RA:
    - Chỉ trả về danh sách 5 queries.
    - Mỗi query một dòng.
    - Không giải thích gì thêm.
    - Các query phải chứa Ticker và Năm để định vị chính xác.
    """

    try:
        res = await openai_complete_if_cache(prompt, model=settings.GPT_MODEL, temperature=0.1, max_tokens=1024)
        
        lines = [l.strip() for l in res.splitlines() if l.strip()]
        clean_queries = []
        for l in lines:
            clean_q = re.sub(r'^[\d\-\.\)\*]+\s+', '', l)
            if len(clean_q) > 5:
                clean_queries.append(clean_q)
        
        if not clean_queries:
            clean_queries = [
                f"{question} {year}",
                f"Báo cáo tài chính {ticker} {year}",
                f"Thuyết minh {question} {year}"
            ]
            
        return clean_queries[:5]

    except Exception as e:
        print(f"[Query Gen Error] {e}", file=sys.stderr)
        return [question, f"Báo cáo tài chính {ticker} {year}"]

async def refine_answer(question, raw_context):
    if len(raw_context) < 100: 
        return "Dữ liệu hiện không có trong báo cáo."

    prompt = f"""
    ROLE: Bạn là Chuyên gia Kiểm toán (Auditor) đang soát xét Báo cáo Tài chính tại Việt Nam.

    Bạn PHẢI trả lời dựa CHỈ trên CONTEXT OCR bên dưới. Tuyệt đối không bịa/đoán.

    ====================
    CONTEXT (OCR):
    {raw_context}
    ====================

    QUESTION:
    {question}

    QUY TẮC BẮT BUỘC (QUALITY GATES):
    1) KHÔNG ĐƯỢC BỊA SỐ:
    - Mọi chữ số trong câu trả lời phải xuất hiện nguyên văn trong CONTEXT
        (ngoại lệ duy nhất xem mục "PHÉP TÍNH RẤT HẠN CHẾ").
    - Nếu không tìm thấy số liệu đáp ứng đúng chỉ tiêu + đúng kỳ + đúng đơn vị -> phải trả lời "không có dữ liệu".

    2) KHỚP ĐÚNG CHỈ TIÊU (LABEL):
    - Khi trả lời, PHẢI dùng đúng tên chỉ tiêu (Label) y hệt như trong CONTEXT (copy nguyên văn).
    - Không được rút gọn/đổi tên chỉ tiêu theo cách của bạn.

    3) KHỚP ĐÚNG ĐỐI TƯỢNG:
    - Nếu trong CONTEXT không có tên/ticker/ngữ cảnh nhận diện đúng công ty/ngân hàng được hỏi -> coi như không đủ dữ liệu.

    4) CHỌN ĐÚNG BẢNG & KỲ:
    - "tại ngày / ngày dd/mm/yyyy" -> ưu tiên Bảng cân đối kế toán.
    - "trong X tháng / lũy kế / 9 tháng / quý" -> ưu tiên Báo cáo KQKD hoặc LCTT tùy chỉ tiêu.
    - Luôn chọn đúng cột thời gian tương ứng (kỳ này/kỳ trước; lũy kế/quý).

    5) ĐƠN VỊ:
    - Giữ nguyên đơn vị tiền tệ đúng theo header của bảng (ví dụ: "triệu đồng", "tỷ đồng", "VND", "đồng").
    - Không tự đổi đơn vị.

    6) BẪY LCTT:
    - Với "Lưu chuyển tiền thuần từ hoạt động kinh doanh": nếu xuất hiện 2 lần, PHẢI chọn dòng tổng hợp cuối cùng (thường ở cuối mục, in đậm, gần các mục I/II/III).
    - Không tự cộng/trừ để ra số LCTT.

    7) XUNG ĐỘT NHIỀU SỐ (CỰC QUAN TRỌNG):
    Nếu có nhiều giá trị giống nhau/na ná cho cùng câu hỏi, áp dụng thứ tự ưu tiên:
    (a) Dòng có LABEL khớp sát nhất với câu hỏi (ưu tiên trùng cụm từ chính).
    (b) Dòng nằm trong bảng có "Mã số" / cấu trúc cột rõ ràng.
    (c) Dòng có ngày/kỳ khớp chính xác với QUESTION.
    Nếu sau ưu tiên (a)(b)(c) vẫn còn mơ hồ -> trả lời "không có dữ liệu" (để tránh chọn sai).

    8) QUY TẮC SỐ ÂM:
    Trong báo cáo tài chính Việt Nam, các số âm được cho vào trong ngoặc như (398.631.979.587) VND, thì phải trả lời là -398.631.979.587 VND

    CHUẨN HOÁ ĐỊNH DẠNG SỐ:
    - Dùng dấu chấm (.) phân cách hàng nghìn: ví dụ 876.226.156
    - Nếu OCR dùng dấu phẩy (,) như 876,226,156 thì hiểu là phân cách hàng nghìn và đổi thành 876.226.156
    - KHÔNG được tự ý làm tròn.
    - KHÔNG thêm dấu chấm ở cuối câu.
    - SỐ ÂM:
    - Nếu số trong CONTEXT ở dạng ngoặc (9.440.425) thì trả lời dạng -9.440.425
    - Nếu số đã có dấu "-" thì giữ nguyên.

    PHÉP TÍNH RẤT HẠN CHẾ (chỉ dùng khi đáp án KHÔNG có sẵn dưới dạng 1 dòng trong bảng):
    - CHỈ được tính khi:
    (1) QUESTION hỏi đúng chỉ tiêu "Lợi nhuận kế toán sau thuế thu nhập doanh nghiệp" hoặc "Lợi nhuận sau thuế"
    (2) CONTEXT có đầy đủ 2 số cùng kỳ, cùng đơn vị:
        "Lợi nhuận kế toán trước thuế" và "Chi phí thuế thu nhập doanh nghiệp"
    (3) Không có dòng "Lợi nhuận ... sau thuế" tương ứng trong CONTEXT
    - Khi tính, phải ghi cực ngắn trong cùng câu: "(tính từ A - B)" và A/B phải là số có trong CONTEXT.
    - Ngoài trường hợp này: TUYỆT ĐỐI KHÔNG TÍNH TOÁN.

    YÊU CẦU TRẢ LỜI (OUTPUT):
    - Chỉ trả lời 1 câu duy nhất, không xuống dòng, không bullet, không "giải thích", không "tham chiếu", không ký hiệu [1].
    - Mẫu khi CÓ số:
    "<LABEL> của <đối tượng như trong QUESTION> <kỳ trong QUESTION> là/đạt <SỐ> <ĐƠN VỊ>"
    - Mẫu khi KHÔNG CÓ số:
    "<chỉ tiêu trong QUESTION> của <đối tượng trong QUESTION> <kỳ trong QUESTION> hiện không có trong báo cáo"
    """

    response = await openai_complete_if_cache(
        prompt, 
        model_name=settings.REASONER_MODEL, 
        temperature=0.1,
        max_tokens=8192 
    )
    
    final_ans = remove_think_tag(response)
    
    return final_ans

async def query_func(placeholder, question: str, mode: str):
    ticker, year, quarter = await extract_metadata_smart(question)
    
    print(f"\n>>> RAG ROUTING: {ticker} | {quarter}/{year}", file=sys.stderr) 
    
    rag = get_rag_engine(ticker, year, quarter)
    
    if not os.path.exists(rag.working_dir):
        return "", f"Xin lỗi, hiện tại hệ thống chưa có dữ liệu báo cáo của {ticker} vào {quarter}/{year}."

    await rag.initialize_storages()
    
    search_list = [question]
    expanded = await generate_search_queries(question, ticker, year, quarter)
    search_list.extend(expanded)
    
    tasks = [rag.aquery(q, param=QueryParam(mode=mode, top_k=100)) for q in search_list]
    responses = await asyncio.gather(*tasks) 
    
    valid_contexts = [r for r in responses if isinstance(r, str) and len(r) > 200]

    MAX_CTX = 30   
    valid_contexts = valid_contexts[:MAX_CTX]

    final_raw_context = "\n\n---\n\n".join(valid_contexts)

    final_ans = await refine_answer(question, final_raw_context)

    return valid_contexts, final_ans
