# sudo apt-get install -y python3-pyaudio python3-pip python3-venv python3-dev
# sudo apt install sox libsox-fmt-all
# pip install typing_extensions
# pip install vosk numpy webrtcvad sox
# 

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
from scipy.signal import resample_poly
import argparse

# Xử lý tham số đầu vào
parser = argparse.ArgumentParser(description="Speech Recognition using Vosk")
parser.add_argument("--input", type=str, help="Path to input WAV file")
args = parser.parse_args()

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

def resample_audio(audio_data):
    # Convert bytes to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    # Resample từ 48000 Hz xuống 16000 Hz
    audio_resampled = resample_poly(audio_np, up=1, down=3)
    
    # Resample từ 48000 xuong 16000
    #transformer = sox.Transformer()
    #transformer.set_output_format(rate=SAMPLE_RATE_VAD)
    #return transformer.build_array(input_array=audio_np, sample_rate_in=SAMPLE_RATE_ORIGINAL)
    return np.int16(audio_resampled)

silent_counter = 0

if args.input:
    # Nhận diện từ file WAV
    print(f"🔍 Đang nhận diện file: {args.input}")
    with open(args.input, "rb") as f:
        f.seek(44)  # Bỏ header WAV
        audio_data = f.read()
    
    if recognizer.AcceptWaveform(audio_data):
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        print("📄 Kết quả từ file:", text)
    else:
        print("⚠️ Không nhận diện được nội dung từ file.")

else:
    # Nhận diện từ micro (giữ nguyên code cũ)
    print("🎤 Đang sử dụng micro để nhận diện...")
    usb_device_index = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if "USB Audio" in device_info["name"]:  # Tìm thiết bị USB Audio
            usb_device_index = i
            print(f"🎤 Đã tìm thấy thiết bị USB Audio: {device_info['name']} (index {i})")
            break

    if usb_device_index is None:
        raise Exception("Không tìm thấy thiết bị USB Audio!")

    # Mở luồng âm thanh từ thiết bị USB
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE_ORIGINAL,
                    input=True, input_device_index=usb_device_index, frames_per_buffer=FRAME_SIZE)

    try:
        with open(output_file, "a") as f:
            while True:
                data = stream.read(FRAME_SIZE, exception_on_overflow=False)
                data_resampled = resample_audio(data).tobytes()
                is_speech = vad.is_speech(data_resampled, SAMPLE_RATE_VAD)
                if is_speech:
                    print("🎤 Đang lắng nghe...")
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "")
                        print("Kết quả nhận dạng:", text)
                        if text:
                            f.write(text + "\n")
                            f.flush()
                else:
                    silent_counter += 1
                    if silent_counter >= 600:
                        print("🤫 Người dùng đã ngừng nói")
                        break
    except KeyboardInterrupt:
        print("\nNhận dạng kết thúc.")
        stream.stop_stream()
        stream.close()
        p.terminate()