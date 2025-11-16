import os
import time
import numpy as np
import serial
from pydub import AudioSegment
import simpleaudio as sa

# ---------- CONFIG ----------
COM_PORT   = "COM5"                # <--- CHANGE THIS
BAUD       = 115200
SONG_FOLDER = "songs"              # folder containing  mp3
SONG_FILE   = "song1.mp3"          # <--- CHANGE TO  DEMO SONG
MODEL_PATH  = "models/haptic.tflite"  # <--- PATH TO  .tflite
FRAME_MS    = 100                  # frame length in ms (tune to  model)
# -----------------------------

# ---- TFLite setup ----
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # from tf if installed

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Loaded TFLite model:", MODEL_PATH)
print("Input details:", input_details)
print("Output details:", output_details)


def open_serial():
    ser = serial.Serial(COM_PORT, BAUD, timeout=1)
    time.sleep(2)  # Nano resets on connect
    print("Serial open on", COM_PORT)
    return ser


def load_song():
    song_path = os.path.join(SONG_FOLDER, SONG_FILE)
    audio = AudioSegment.from_mp3(song_path)
    print(f"Loaded song: {song_path}")
    print(f"Duration: {len(audio)/1000:.2f} s, sample rate: {audio.frame_rate} Hz")
    return audio


# ---------- YOU MAY NEED TO ADAPT THESE TWO FUNCTIONS ----------

def preprocess_frame(frame_samples, sample_rate):
    """
    Convert raw 1D audio frame into the input tensor  model expects.

    This is a generic example:
    - normalize
    - reshape to [1, N] if model expects (1, N)

    Replace with  actual preprocessing from the notebook if different.
    """
    x = frame_samples.astype('float32')
    x /= (np.max(np.abs(x)) + 1e-6)

    # Example: model expects shape (1, N)
    in_info = input_details[0]
    target_shape = in_info['shape']  # e.g. [1, 1024]
    # Simple case: just crop/pad to match length
    N = target_shape[-1]
    if x.shape[0] < N:
        pad = np.zeros(N - x.shape[0], dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)
    else:
        x = x[:N]

    x = x.reshape(target_shape)
    return x


def run_model_on_frame(frame_samples, sample_rate):
    """
    Run TFLite inference on one frame and return (y_low, y_mid, y_high).
    Assumes model output is shaped [1, 3] or similar.

    Adapt the output indexing if  model is different.
    """
    x = preprocess_frame(frame_samples, sample_rate)

    # Set input tensor
    idx_in = input_details[0]['index']
    x = x.astype(input_details[0]['dtype'])
    interpreter.set_tensor(idx_in, x)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    idx_out = output_details[0]['index']
    y = interpreter.get_tensor(idx_out)

    # Assume y shape is [1, 3] => [low, mid, high]
    y = np.squeeze(y)  # shape (3,)
    y_low, y_mid, y_high = float(y[0]), float(y[1]), float(y[2])

    return y_low, y_mid, y_high

# ---------------------------------------------------------------


def to_pwm(y, min_val=0.0, max_val=1.0):
    """Map arbitrary float to [0, 255] for PWM."""
    v = (y - min_val) / (max_val - min_val + 1e-6)
    v = np.clip(v, 0.0, 1.0)
    return int(v * 255)


def main():
    ser = open_serial()
    audio = load_song()

    # Start playback (non-blocking)
    play_obj = audio.play()
    print("Playback started, streaming haptics...")

    sample_rate = audio.frame_rate
    frame_samples_count = int(sample_rate * FRAME_MS / 1000.0)

    # Convert to numpy mono
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)

    total_ms = len(audio)
    t_ms = 0
    start_time = time.time()

    while t_ms + FRAME_MS <= total_ms:
        # Indices in samples
        start_sample = int(t_ms * sample_rate / 1000.0)
        end_sample   = start_sample + frame_samples_count
        frame = samples[start_sample:end_sample]

        if len(frame) < frame_samples_count:
            break

        # ---- TFLite inference ----
        y_low, y_mid, y_high = run_model_on_frame(frame, sample_rate)

        # Map to PWM 0–255
        # You can tune min/max based on  model's output range
        L_pwm = to_pwm(y_low, 0.0, 1.0)
        M_pwm = to_pwm(y_mid, 0.0, 1.0)
        H_pwm = to_pwm(y_high, 0.0, 1.0)  # Arduino will ignore H for now

        # Send "L M H\n" to Arduino
        line = f"{L_pwm} {M_pwm} {H_pwm}\n"
        ser.write(line.encode("ascii"))
        # print("Sent:", line.strip())  # uncomment for debug

        # Keep timing roughly real-time with audio
        t_ms += FRAME_MS
        target_time = start_time + t_ms / 1000.0
        now = time.time()
        sleep_time = target_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)

        if not play_obj.is_playing():
            break

    print("Streaming done. Turning motors off.")
    ser.write(b"0 0 0\n")
    time.sleep(0.1)
    ser.close()


if __name__ == "__main__":
    main()
