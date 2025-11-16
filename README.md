# Haptica
ML model to translate music into haptics with Arduino Nano and coin vibrator. 

# Haptica

ML model to translate music into haptics using an Arduino Nano and coin vibration motors.

---

## 1. Why Haptica?

Do you listen to music blasting on speakers for the full-body vibe?  
Or with earphones to disappear into your own world?

There is no single way to experience music — and there is no single way Deaf and hard-of-hearing (DHH) people experience it either:

- Some have residual hearing and enjoy music through speakers or headphones.
- Some prefer feeling loud music as tactile vibrations.
- Some prefer quiet.
- Some enjoy signed music. Some do not.

DHH people are not a monolith; they are individuals with their own preferences and hearing levels.

**Haptica** is a small prototype that explores one idea:

> If sound can be felt through our skin, can we translate parts of music into vibration patterns so that transitions and energy changes are *felt*, not just heard?

The goal is not to “fix” anything or claim a universal solution — it is to learn, experiment, and move a little closer to making music accessible in more forms.

---

## 2. Project Overview

**High-level flow:**

1. Take music as input.
2. Extract compact audio features.
3. Run them through a tiny ML model on a microcontroller.
4. Drive coin vibration motors so the user can feel changes in the music.

**Key components:**

- **ML model**
  - Lightweight audio model using **depthwise-separable convolutions**.
  - Trained in Python / Jupyter.
  - Exported to a `.tflite` file for deployment.

- **Hardware**
  - **Arduino Nano–class board** (chosen for low power and tight memory).
  - **Coin vibration motor(s)** as the haptic output.
  - Simple vibration patterns (on/off, pulses, intensity changes) mapped from model predictions.

- **Deployment**
  - Model trained offline → converted to TensorFlow Lite → used either:
    - in a Python desktop demo, or
    - on the Nano via Arduino + TFLite for Microcontrollers (using a `model.h` C array if needed).

This is a **prototype** for a haptic music translator, not a medical device.

---

## 3. Repository Contents (current snapshot)

Depending on the latest commit, you should see some or all of:

- `*.ipynb` – Jupyter notebook(s) for data exploration and model training.
- `python/` – Python scripts, `.tflite` model, and a `songs/` folder for test audio.
- `arduino/` or `.ino` file – Arduino IDE sketch for the Nano + coin motor control.
- `dataset/` – Training / test data for the ML model.
- `output/` – Notebook outputs (plots, logs, saved artifacts).

The structure may be refined later (e.g., splitting into `notebooks/`, `models/`, `edge_runtime/`, etc.), but this README already captures the core purpose and components.

---


## 4. Notes & Future Directions

- This is an early-stage prototype and has not been co-designed with DHH users yet.
- Haptic patterns are simple and can be expanded into richer “vocabularies” for rhythm, sections, or instruments.
- Future ideas:
  - more motors and spatial patterns,
  - user-tunable intensity and mapping,
  - collaboration with Deaf and hard-of-hearing communities for feedback and design.

Accessibility is not about one perfect interface — it is about adding more ways for people to choose how they want to experience the world. Haptica is one small experiment in that direction.
