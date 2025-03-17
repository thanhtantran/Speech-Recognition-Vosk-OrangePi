# sudo apt-get install pip
# sudo apt-get install -y python3-pyaudio
# sudo pip3 install vosk
# sudo apt install sox libsox-fmt-all

import os
import sys
import json
import contextlib
import pyaudio
import io
import sox
import numpy as np
import pyaudio
import webrtcvad
from vosk import Model, KaldiRecognizer

SAMPLE_RATE_ORIGINAL = 48000
SAMPLE_RATE_VAD = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30  # 10, 20 hoặc 30 ms
FRAME_SIZE = int(SAMPLE_RATE_VAD / 1000 * FRAME_DURATION_MS)
FORMAT = pyaudio.paInt16

# Path to the Vosk model
model_path = "models/vosk-model-vn-0.4/"
if not os.path.exists(model_path):
    print(f"Model '{model_path}' was not found. Please check the path.")
    exit(1)
    
# File lưu kết quả
output_file = "recognized_text.txt"

model = Model(model_path) # Cung cấp đường dẫn tới model vosk
recognizer = KaldiRecognizer(model, SAMPLE_RATE_VAD)
vad = webrtcvad.Vad(3)

p = pyaudio.PyAudio()

# Tìm đúng thiết bị USB Audio Device
usb_device_index = None
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if "USB Audio" in device_info["name"]:  # Tên chứa "USB Audio"
        usb_device_index = i
        print(f"🎤 Đã tìm thấy thiết bị USB Audio: {device_info['name']} (index {i})")
        break

if usb_device_index is None:
    raise Exception("Không tìm thấy thiết bị USB Audio!")

# Mở luồng âm thanh từ thiết bị USB
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE_ORIGINAL, input=True, input_device_index=usb_device_index, frames_per_buffer=FRAME_SIZE)

def resample_audio(audio_data):
    # Convert bytes to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    # Resample từ 48000 xuong 16000
    transformer = sox.Transformer()
    transformer.set_output_format(rate=SAMPLE_RATE_VAD)
    return transformer.build_array(input_array=audio_np, sample_rate_in=SAMPLE_RATE_ORIGINAL)

silent_counter = 0

try:
    with open(output_file, "a") as f:  # Mở file ở chế độ "append" để ghi tiếp nối
        while True:
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            data_resampled = resample_audio(data).tobytes()
            is_speech = vad.is_speech(data_resampled, SAMPLE_RATE_VAD)
            if is_speech:
                print("🎤 Đang lắng nghe...")
                # Nhận diện giọng nói với KaldiRecognizer
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result()) # Kết quả dạng JSON
                    text = result.get("text", "")  # Lấy nội dung text
                    print("Kết quả nhận dạng:",result)
                    if text:
                        print(text)  # Hiển thị trên màn hình
                        f.write(text + "\n")  # Ghi tiếp vào file
                        f.flush()  # Đảm bảo nội dung được ghi ngay
            else:
                silent_counter += 1
                if silent_counter >= 300:  # Khoảng 2 giây (nếu FRAME_SIZE ~ 20ms)
                    print("🤫 Người dùng đã ngừng nói")
                    break
except KeyboardInterrupt:
    print("\nNhận dạng kết thúc.")
    stream.stop_stream()
    stream.close()
    p.terminate()