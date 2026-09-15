import requests

try:
    img_path = r"C:\Users\Samuel\.cache\kagglehub\datasets\naureenmohammad\mmu-iris-dataset\versions\1\MMU-Iris-Database\1\left\aeval4.bmp"
    with open(img_path, 'rb') as f:
        files = {'photo': ('aeval4.bmp', f, 'image/bmp')}
        response = requests.post('http://127.0.0.1:7860/identify', files=files)
        
    print(f"Status: {response.status_code}")
    if response.status_code == 500:
        print("Response text containing error:")
        print(response.text)
except Exception as e:
    print(f"Failed to make request: {e}")
