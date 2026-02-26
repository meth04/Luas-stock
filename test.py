from openai import OpenAI

# Khởi tạo client và trỏ base_url về cliProxyAPI của bạn
client = OpenAI(
    base_url="http://127.0.0.1:8317/v1",
    api_key="sk-my-key-is-empty" 
)

try:
    print("Đang gửi yêu cầu kiểm tra...")
    response = client.chat.completions.create(
        model="kimi-k2-thinking",
        messages=[
            {"role": "user", "content": "Nghiên cứu kỹ và trả lời cho tôi biết bạn là ai và tương lai của trí tuệ nhân tạo sẽ thế nào?"}
        ]
    )
    print("\n✅ API HOẠT ĐỘNG THÀNH CÔNG!")
    print("Phản hồi:", response.choices[0].message.content.strip())

except Exception as e:
    print(f"\n❌ LỖI: Cấu hình hoặc kết nối có vấn đề.\nChi tiết: {e}")