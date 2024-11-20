import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_KEY"])

def load_llm(model_name="gemini-1.5-flash-8b"):
    model = genai.GenerativeModel(model_name=model_name)
    return model

def extract_lyrics(text):
    start_marker = "[START]"
    end_marker = "[END]"

    start_index = text.find(start_marker) + len(start_marker)
    end_index = text.find(end_marker)

    # Get the content in between and strip leading/trailing whitespace
    content_between = text[start_index:end_index].strip()

    return content_between

def generate_corrections(llm_model, text_result):
    prompt = 'Đây là lời bài hát được chép lại từ bài hát, nên nó không chính xác lắm. Có những lỗi sai như ghép 2 từ thành 1, và thiếu dấu câu. Hãy viết lại, sửa lỗi nếu cần, thêm dấu câu, và giữ nguyên format ' \
        + '\n [START] \n'   \
        + text_result       \
        + '\n [END]'        \

    response = llm_model.generate_content(prompt)

    text_result = extract_lyrics(response.text)

    return text_result

if __name__ == "__main__":
    model = load_llm()
    text_result = \
    '''
    [00:29.98] Bài  Hát  Cuối  Ca  Sĩ  &  Đàn  Ông  HồngNgười  ra  điEm  theo  anh  về  Quảng  Ngãi  một  lần  được  không 
    [00:58.78]  Qua  cầu  Bổ  Ly  Trưa  nắng  hạ  là  nước  long  lanh  Ngồi  kề  bên  anh  Nôn  nao  ngắm  ngó  Thương  sao  đất  lành  Đi  qua  nghĩa  hạnh  Qua  sơn  tình  tư  nghĩa  mộng  mơ  Chiều  bến  Tam  Thương  Thương  sông 
    [01:28.78]  Trà  hàng  ghế  che  bờ  Đẹp  tình  ba  tơ  Thương  sông  núi  Minh  Long  Sơn  Hà  Em  theo  anh  qua  những  con  đường  Mang  bao  di  tích  đời  đời  Quê  tôi  mồ  đực  đây  rồi  Chiều  quê 
    [01:58.08]  chiều  quê  Thương  ai  ai  cấy  mà  non  Thương  anh  Thương  anh  Quảng  Ngãi  em  về  Sơn  tay  giữ  mãi  câu  thề  Trà  Bồng  dịu  dàng  nắng  trong  Bình  Sơn  Yêu  sao  tiếng  hát  nhớ  nhau  Thương  anh  Em  về  cùng 
    [02:27.68]  anh  Qua  bao  năm  xuôi  ngược  Nay  trở  về  Quảng  Ngãi  quê  tôi  Qua  Lý  Sơn  xưa  Nay  thuyền  bè  sum  đúc  đôi  bờ  Đức  Phố  mẹ  tôi  Còn  đứng  đó  trong  đợi  con  về  Quảng  Ngãi  mẹ  tôi 
    [02:56.92]  Còn  đứng  đó  trong  đợi  con 
    [03:00.20]  vềNgười  ơi  em  có  biết  Bao  nhi 
    [03:44.40]  một  lần  được  không  Qua  cầu  Bổ  Lũy  Trưa  nắng  hạ  làng  nước  long  lanh  Ngồi  kề  bên  anh  Nôn  nao  ngắm  ngó  Thương  sao  đất  lành  Đi  qua  Nghĩa  Hành  Qua  Sơn  tình  tư  nghĩa  mộng  mơ  Chiều  bến  tang 
    [04:14.38]  thương  Thương  sông  Trà  Hàng  ghế  che  bờ  Đẹp  tình  ba  cơ  Thương  sông  núi  Mênh  long  sơn  hà  Em  theo  anh  qua  những  con  đường  Mang  bao  di  tích  đời  đời  Quê  tôi  mồ  đực  đây  rồi 
    [04:44.02]  Chiều  quê  chiều  quê  Thương  ai  ai  cấy  mà  non  Thương  anh  thương  anh  Quảng  Ngãi  em  về  Sơn  tay  giữ  mãi  câu  thề  Trà  Bồng  dịu  dàng  nắng  trongình  Sơn  Yêu  sao  tiếng  hát  nhớ  nhau  Thương  anh  thương  anh 
    [05:13.92]  Em  về  cùng  anh  Qua  bao  năm  xuôi  ngược  Nay  trở  về  Quảng  Ngãi  quê  tôi  Qua  Lý  Sơn  xưa  Nay  Thuyền  Bè  sum  đúc  đôi  bờ  Đức  Phố  mẹ  tôi 
    '''
    
    text_result = generate_corrections(model, text_result)
    print(text_result)