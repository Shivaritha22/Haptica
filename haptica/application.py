"""
Haptic player:
- Reads an MP3 on the laptop
- Runs a TFLite model to get [Y_low, Y_mid, Y_high] per frame
- Sends 3 intensity values (0..255) to Arduino Nano over Serial
- Logs model outputs to a text file like:
    # song: song.mp3
    f0 = [ylow, ymid, yhigh]
    f1 = [ylow, ymid, yhigh]
"""

import time
import os
import numpy as np
import serial
import librosa

# ========= USER CONFIG (EDIT THESE) =========

MP3_PATH = "C:\Users\shiva\Documents\VScode projects\haptica\song.mp3"              # path to demo song
TFLITE_MODEL_PATH = "model.tflite" #  trained TFLite model

# Serial port where the Nano shows up:
#   Windows: "COM5"
#   Linux:   "/dev/ttyACM0"
#   macOS:   "/dev/cu.usbmodemXXXX"
SERIAL_PORT = "COM5"
BAUD_RATE = 115200

# Audio / feature settings (match  training setup)
TARGET_SR = 16000       # sample rate used for training
FRAME_HOP_SEC = 0.05    # 50 ms per frame
N_MELS = 32             # mel bins used during training
WIN_FRAMES = 16         # frames per model input window

# Log file for Y_low, Y_mid, Y_high
LOG_PATH = "haptic_log.txt"

# ============================================

# Try lightweight tflite runtime first, then fall back to full TF
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


def load_audio(path: str, sr: int = TARGET_SR):
    """Load mono audio at target sample rate."""
    audio, sr = librosa.load(path, sr=sr, mono=True)
    return audio, sr


def compute_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a log-mel spectrogram.
    IMPORTANT:
      This should match training notebook's preprocessing.
    """
    hop_length = int(FRAME_HOP_SEC * sr)  # samples between frames
    n_fft = 1024

    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MELS,
        power=2.0,
    )  # (n_mels, n_frames)

    S_db = librosa.power_to_db(S, ref=np.max)

    # Simple 0..1 normalization 
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm  # (n_mels, n_frames)


def make_model_patches(mel_spec: np.ndarray, win_frames: int) -> np.ndarray:
    """
    Turn mel spectrogram into overlapping patches for model input.

    mel_spec: (n_mels, n_frames)
    returns: (num_windows, win_frames, n_mels, 1)
    """
    n_mels, n_frames = mel_spec.shape
    if n_frames < win_frames:
        raise ValueError(f"Audio too short for WIN_FRAMES = {win_frames}")

    patches = []
    # Slide a window along time axis with stride 1 frame
    for start in range(0, n_frames - win_frames + 1):
        end = start + win_frames
        patch = mel_spec[:, start:end]   # (n_mels, win_frames)
        patch = patch.T                  # (win_frames, n_mels)
        patches.append(patch)

    patches = np.array(patches, dtype=np.float32)          # (num_windows, win_frames, n_mels)
    patches = np.expand_dims(patches, axis=-1)             # (num_windows, win_frames, n_mels, 1)
    return patches


def load_interpreter(model_path: str):
    """Load TFLite model and return interpreter + IO details."""
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("=== TFLite model loaded ===")
    print("Input details:", input_details)
    print("Output details:", output_details)
    print("===========================")

    return interpreter, input_details, output_details


def run_model_on_patches(
    interpreter,
    input_details,
    output_details,
    patches: np.ndarray
) -> np.ndarray:
    """
    Run TFLite model on all patches.
    Returns: (num_windows, 3) array with [Y_low, Y_mid, Y_high] per patch.
    """
    num_windows = patches.shape[0]
    outputs = []

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    input_dtype = input_details[0]['dtype']
    output_dtype = output_details[0]['dtype']
    in_scale, in_zero = input_details[0]['quantization']
    out_scale, out_zero = output_details[0]['quantization']

    for i in range(num_windows):
        x = patches[i:i+1]  # (1, win_frames, n_mels, 1)

        # Handle quantized or float input
        if input_dtype == np.float32:
            x_in = x.astype(np.float32)
        else:
            # Quantized: float -> int8/uint8
            x_in = x / in_scale + in_zero
            x_in = np.clip(np.round(x_in), 0, 255).astype(input_dtype)

        interpreter.set_tensor(input_index, x_in)
        interpreter.invoke()

        y = interpreter.get_tensor(output_index)[0]  # e.g. shape (3,)

        # Dequantize if needed
        if output_dtype != np.float32:
            y = (y.astype(np.float32) - out_zero) * out_scale

        outputs.append(y)

    outputs = np.array(outputs)  # (num_windows, 3)
    return outputs


def to_pwm_values(outputs: np.ndarray) -> np.ndarray:
    """
    Map model outputs to integer PWM 0..255 per channel.

    """
    outputs = np.clip(outputs, 0.0, 1.0)
    pwm = (outputs * 255.0).astype(np.int16)
    return pwm  # (num_windows, 3) ints


def save_haptic_log_text(raw_outputs: np.ndarray, song_name: str, path: str = LOG_PATH):
    """
    Save model outputs to a text file in the format:
    # song: <song_name>
    f0 = [ylow, ymid, yhigh]
    f1 = [ylow, ymid, yhigh]
    ...
    """
    num_frames = raw_outputs.shape[0]
    with open(path, "w") as f:
        f.write(f"# song: {song_name}\n")
        f.write("# format: f<index> = [y_low, y_mid, y_high]\n\n")
        for i in range(num_frames):
            y_low, y_mid, y_high = raw_outputs[i]
            line = f"f{i} = [{float(y_low):.6f}, {float(y_mid):.6f}, {float(y_high):.6f}]\n"
            f.write(line)
    print(f"Saved haptic log to {path}")


def stream_to_nano(pwm_vals: np.ndarray, frame_hop_sec: float):
    """
    Stream precomputed [L,M,H] triples to Arduino Nano over serial.
    One triple per frame_hop_sec seconds.

    Arduino should expect lines formatted as: "L,M,H\n"
    with 0 <= L, M, H <= 255.
    """
    print(f"Opening serial port {SERIAL_PORT} @ {BAUD_RATE}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    # Give Arduino a moment to reset after opening serial
    time.sleep(2.0)
    print("Streaming PWM values to Nano...")

    start_time = time.time()
    num_frames = len(pwm_vals)

    for i, (v_low, v_mid, v_high) in enumerate(pwm_vals):
        target_time = start_time + i * frame_hop_sec

        # Wait until it's time for this frame
        now = time.time()
        sleep_time = target_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)

        line = f"{int(v_low)},{int(v_mid)},{int(v_high)}\n"
        ser.write(line.encode("ascii"))

        # Optional debug print
        if i % 50 == 0:
            print(f"Frame {i}/{num_frames}: {line.strip()}")

    ser.close()
    print("Done streaming to Nano.")


def main():
    # 1) Load audio from mp3
    print("Loading audio from:", MP3_PATH)
    audio, sr = load_audio(MP3_PATH, sr=TARGET_SR)
    duration_sec = len(audio) / sr
    print(f"Audio length: {duration_sec:.1f} s, sample rate: {sr} Hz")

    # 2) Compute mel spectrogram
    print("Computing mel spectrogram...")
    mel_spec = compute_mel_spectrogram(audio, sr)
    print("Mel spec shape (n_mels, n_frames):", mel_spec.shape)

    # 3) Build model input patches
    print("Creating model input patches...")
    patches = make_model_patches(mel_spec, win_frames=WIN_FRAMES)
    print("Patches shape (num_windows, win_frames, n_mels, 1):", patches.shape)

    # 4) Load TFLite model and run inference
    print("Loading TFLite model...")
    interpreter, in_details, out_details = load_interpreter(TFLITE_MODEL_PATH)

    print("Running inference on all patches...")
    raw_outputs = run_model_on_patches(interpreter, in_details, out_details, patches)
    print("Raw outputs shape:", raw_outputs.shape)

    # 5) Save log of Y_low, Y_mid, Y_high with song name in header
    song_name = os.path.basename(MP3_PATH)
    save_haptic_log_text(raw_outputs, song_name=song_name, path=LOG_PATH)

    # 6) Convert model outputs to PWM values (0..255)
    pwm_vals = to_pwm_values(raw_outputs)
    print("PWM values shape:", pwm_vals.shape)

    # 7) Stream to Arduino Nano over serial
    print("\n>>> At this point, start playing the SAME song on your laptop (VLC/Spotify/etc.)")
    input("Press Enter here when you’re ready to stream haptics... ")

    stream_to_nano(pwm_vals, frame_hop_sec=FRAME_HOP_SEC)


if __name__ == "__main__":
    main()
