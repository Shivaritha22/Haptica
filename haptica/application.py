import time
import os
import numpy as np
import serial
from pydub import AudioSegment
import simpleaudio as sa

# ---------- CONFIG ----------
COM_PORT = "COM5"      # CHANGE to Nano port
BAUD = 115200
SONG_FOLDER = "songs"  # CHANGE if needed
SONG_FILE   = "song1.mp3"  # the mp3 you want to demo
FRAME_MS    = 100      # frame length in milliseconds 
# ----------------------------

def open_serial():
  ser = serial.Serial(COM_PORT, BAUD, timeout=1)
  time.sleep(2)  # allow Nano to reset
  print("Serial open on", COM_PORT)
  return ser

def load_song():
  song_path = os.path.join(SONG_FOLDER, SONG_FILE)
  audio = AudioSegment.from_mp3(song_path)
  print(f"Loaded song: {song_path}")
  print(f"Duration: {len(audio)/1000:.2f} s, sample rate: {audio.frame_rate} Hz")
  return audio

# ---- PLACEHOLDER:  TFLite inference ----
# You already have this in  notebook. Just wrap that logic here.
def run_model_on_frame(frame_samples, sample_rate):
  """
  frame_samples: 1D np.array of audio samples (float32 or int16)
  sample_rate:   int

  Returns:
      y_low, y_mid, y_high (floats)
  """
  # TODO: replace this with  actual model inference.
  # For now, just fake something based on energy in the frame for sanity.
  energy = np.mean(frame_samples.astype(np.float32)**2)
  y_low  = energy
  y_mid  = energy * 0.5
  y_high = energy * 0.2
  return y_low, y_mid, y_high

def to_pwm(y, min_val=0.0, max_val=1.0):
  # Map arbitrary float to [0,255]
  v = (y - min_val) / (max_val - min_val + 1e-6)
  v = np.clip(v, 0.0, 1.0)
  return int(v * 255)

def main():
  ser = open_serial()
  audio = load_song()

  # Start playback (non-blocking)
  play_obj = audio.play()
  print("Playback started, beginning haptic streaming...")

  frame_ms = FRAME_MS
  step_ms = FRAME_MS  # hop size; can be smaller than frame_ms if you want overlap

  sample_rate = audio.frame_rate
  frame_samples_count = int(sample_rate * frame_ms / 1000.0)

  # Convert whole song to numpy array (mono)
  samples = np.array(audio.get_array_of_samples())
  if audio.channels == 2:
    samples = samples.reshape((-1, 2)).mean(axis=1)  # stereo -> mono

  total_ms = len(audio)
  t = 0
  start_time = time.time()

  while t + frame_ms <= total_ms:
    # Frame boundaries in samples
    start_sample = int(t * sample_rate / 1000.0)
    end_sample   = start_sample + frame_samples_count
    frame = samples[start_sample:end_sample]

    if len(frame) < frame_samples_count:
      break

    # ---  model inference ---
    y_low, y_mid, y_high = run_model_on_frame(frame, sample_rate)

    # Map to PWM 0-255
    L_pwm = to_pwm(y_low, 0.0, 1.0)
    M_pwm = to_pwm(y_mid, 0.0, 1.0)
    H_pwm = to_pwm(y_high, 0.0, 1.0)  # will be ignored by Arduino for now

    # Send to Arduino as: "L M H\n"
    line = f"{L_pwm} {M_pwm} {H_pwm}\n"
    ser.write(line.encode("ascii"))
    # print("Sent:", line.strip())  # uncomment for debug

    # Try to keep real-time-ish pace
    t += step_ms
    target_time = start_time + t / 1000.0
    now = time.time()
    sleep_time = target_time - now
    if sleep_time > 0:
      time.sleep(sleep_time)

    # Stop if playback finished
    if not play_obj.is_playing():
      break

  print("Done streaming. Turning motors off.")
  ser.write(b"0 0 0\n")
  time.sleep(0.1)
  ser.close()

if __name__ == "__main__":
  main()
