# 😜 Meme Cam

**Meme cam** is a fun real-time AI app that uses your webcam to detect when you **stick your tongue out** 😝 or **close your eyes** 😴 , etc — and flashes matching GIFs or images in a **separate reaction window** as long as you maintain that expression.

It’s powered by [MediaPipe Face Mesh](https://developers.google.com/mediapipe) for landmark detection and [OpenCV](https://opencv.org/) for video processing.

---

## 🎮 Demo

> 👀 The app opens two windows:
>
> * **Meme Cam:** your live webcam feed
> * **Reaction:** shows GIFs or images based on your expression

Press **Q** anytime to close both windows.

---

## 🧬 Features

* 🎥 Real-time webcam tracking
* 😝 Detects **tongue out**
* 😴 Detects **eyes closed**
* 👆 Detects **Monkey pointing up**
* 🤔 Detects **Monkey Thinking**
* 👍 Detects **Monkey thumbs up**
* 🙆‍♂️ Detects **Monkey hands on head**
* 😏 Detects **Smile Stare**
* 🪟 Shows reaction GIFs in a separate window
* ⚙️ Simple setup, no external AI API required

---

## 📦 Requirements

* **Python 3.12.x** (⚠️ `mediapipe` doesn’t support Python 3.13 yet)
* **MediaPipe 0.10.9 ❗**

* A **webcam**
* Works on **Windows**, **macOS**, and **Linux**

---

## 🗂️ Folder Structure

```
meme_cam/
│
├── assets/
│   ├── tongue.gif
│   ├── closed_eyes.gif
│   ├── monkey-pointing.gif
│   ├── monkey-thinking.gif
│   ├── monkey-thumbsup.gif
│   ├── oh_no.gif
│   ├── smile-stare.gif
│
├── output/
│
├── memecam.py
└── README.md
```

---

## 🚀 How To Use 

### 1️⃣ Clone the Repository
Open the Terminal app CMD/Powershell and type: 

```bash
git clone https://github.com/HariSharmaa/MEME-CAM
cd MEME-CAM
```

### 2️⃣ Install Python 3.12

Download from the official site:
🔗 [https://www.python.org/downloads/release/python-3126/](https://www.python.org/downloads/release/python-3126/)
During installation:

* ✅ Check **“Add Python 3.12 to PATH”**
* Then click **Install Now**

Verify:

```bash
python --version
```

---

### 3️⃣ Create a Virtual Environment
Once done typle this command: 
```bash
python -m venv .venv
```

Activate it:

* **Windows:**

  ```bash
  .venv\Scripts\activate
  ```
* **Mac/Linux:**

  ```bash
  source .venv/bin/activate
  ```

---

### 4️⃣ Install Dependencies

With your environment activated:



```bash
pip install opencv-python mediapipe==0.10.9 imageio numpy
```

---

### 5️⃣ Run the App

```bash
python memecam.py
```

Two windows will appear:

* 🎥 `memecam` → your camera feed
* 🪟 `Reaction` → the GIF or image that matches your face action

Press **Q** to quit.

---


## 🧮 Troubleshooting

| Problem                                    | Fix                                                        |
| ------------------------------------------ | ---------------------------------------------------------- |
| ❌ `No matching distribution for mediapipe` | You’re using Python 3.13 — install 3.12.                   |
| 🖼️ GIF not showing                        | Check that the GIFs exist in `assets/` with correct names. |
| 🎥 Camera not opening                      | Make sure no other app (Zoom, Discord, etc.) is using it.  |
| 🪞 Reaction window too small               | Resize it manually or change resolution in code.           |
| ❌ AttributeError                          | Uninstall Mediapipe and reinstall Mediapipe==0.10.9        |

---

## 💡 Future Ideas

* 🔊 Add sound effects for each reaction
* 🧠 Connect GPT or Gemini for smart captions
* 🌐 Launch reactions in a browser tab

---

Feel free to fork, modify, and have fun with it!

---

### ⭐ Don’t forget to star the repo if you like this project ⭐
