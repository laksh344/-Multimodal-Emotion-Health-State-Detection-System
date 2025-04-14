#  Multimodal Emotion & Health State Detection System

A powerful deep learning system that uses **video (facial expressions)**, **audio (speech emotion)**, and **EEG signals** to detect a user's **emotional and mental health state** in real-time. After identifying the emotion, the system provides **personalized AI-generated remedies** using an integrated language model.

> **Repo**: https://github.com/laksh344/-Multimodal-Emotion-Health-State-Detection-System.git

---

## 🚀 Features

- 🎥 **Facial Emotion Recognition** using CNN/Swin Transformers on FER-2013 dataset
- 🔊 **Speech Emotion Recognition** using MFCCs and Spectrograms from RAVDESS dataset
- 🧠 **EEG Emotion Analysis** based on Bonn University EEG dataset
- 🔗 **Hybrid Attention-based Feature Fusion** (video + audio + EEG)
- 🤖 **Deep Learning Architecture**: CNN + LSTM + Dense Layers
- 🧘 **AI Remedy Generator** powered by LLM (Mistral) for mental health suggestions
- ⚡ Mixed-Precision GPU training enabled for fast performance (FP16)

---

## 📁 Datasets Used

| Modality | Dataset          | Description                                              |
|----------|------------------|----------------------------------------------------------|
| EEG      | Bonn University  | Real EEG signals for neurological/mental state analysis |
| Audio    | RAVDESS          | Emotional speech and song dataset (24 actors)           |
| Video    | FER-2013         | 48x48 grayscale facial expression dataset               |

---

## 🎯 Emotion Classes

The model can classify emotions into:

- 😊 Happy
- 😔 Sad
- 😠 Angry
- 😨 Fearful / Anxious
- 😫 Stressed
- 😐 Neutral
- 🧘 Relaxed
- 😞 Depressed

---

## 🧠 Model Architecture

```
[Facial Features]      [Audio Features]       [EEG Features]
       |                      |                        |
    CNN/Swin            MFCC + Conv2D             LSTM/TCN
       |                      |                        |
     Flatten               Flatten                  Flatten
       \_____________________|_______________________/
                             |
               Attention-Based Fusion Layer
                             |
                         Dense Layers
                             |
                          Softmax
```

---

## 🧘 AI-Powered Remedy Suggestion

Once the emotion is detected, the model invokes an LLM (e.g., **Mistral**) to recommend **mental wellness remedies**, such as:

- 💨 Breathing or grounding exercises
- 🎶 Uplifting music or calming nature sounds
- 📔 Journaling prompts for reflection
- 📱 Recommendations for meditation apps
- 📚 Mental health resources or videos

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/laksh344/-Multimodal-Emotion-Health-State-Detection-System.git
cd -Multimodal-Emotion-Health-State-Detection-System
```

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3. Download and Prepare Datasets
- **FER-2013**: [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **RAVDESS**: [RAVDESS Dataset](https://zenodo.org/record/1188976)
- **Bonn EEG**: [Bonn University EEG Dataset](https://epilepsy.uni-freiburg.de)

Ensure datasets are placed in respective folders:
```bash
/data/
    /FER2013/
    /RAVDESS/
    /BONN_EEG/
```

---

## 🔄 Running the Code

### Option 1: Running with Jupyter Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open the main notebook**:
```
swimmmb7.ipynb
```

3. **Run all cells** or execute them sequentially to:
   - Load and preprocess the datasets
   - Train the models (or load pre-trained models)
   - Run the multimodal fusion and emotion detection

### Option 2: Using the Command Line Interface

1. **Train the models**:
```bash
python train_models.py --epochs 50 --batch_size 32 --lr 0.001
```

2. **Run the emotion detection system**:
```bash
python detect_emotion.py --input_video path/to/video.mp4 --input_audio path/to/audio.wav --input_eeg path/to/eeg.csv
```

3. **Use the real-time detection mode**:
```bash
python realtime_detection.py --camera 0 --mic 0 --eeg_device "your_eeg_device"
```

### Option 3: Using the Web Interface

1. **Start the web server**:
```bash
python app.py
```

2. **Access the web interface**: Open your browser and go to:
```
http://localhost:5000
```

3. **Use the interface to**:
   - Upload video/audio/EEG files
   - Use webcam and microphone for real-time detection
   - View emotion detection results and suggested remedies

---

## ⚠️ Important Notes

- GPU is highly recommended for real-time processing
- For EEG integration, compatible devices include:
  - OpenBCI Ganglion/Cyton
  - Muse Headband
  - EMOTIV EPOC/EPOC+
- The system performs best when all three modalities (video, audio, EEG) are available
- Pre-trained models are available in the `/models` directory

---

## 💻 Tech Stack

- **Languages**: Python 3, Markdown
- **Frameworks**: TensorFlow, Keras
- **Libraries**: NumPy, OpenCV, Librosa, SciPy
- **Models**: CNN, Swin Transformer, LSTM, Attention
- **Tools**: Google Colab, Mixed Precision, LLM (Mistral)

---

## 📈 Results

| Modality     | Accuracy   |
|--------------|------------|
| EEG Only     | ~85%       |
| Audio Only   | ~70%       |
| Video Only   | ~75%       |
| Multimodal   | **~90%+**  |

The system shows significant improvement when all three modalities are fused using the hybrid fusion mechanism.

---

## 🔮 Future Work

- [ ] Real-time webcam + mic streaming support
- [ ] Live EEG integration with consumer headsets
- [ ] Web or mobile frontend (React, Flutter)
- [ ] Fine-tuned LLM responses based on personal history

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome!

Feel free to fork this project, improve it, and submit a pull request. ⭐ the repo if you find it helpful.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [Bonn University EEG Dataset](https://epilepsy.uni-freiburg.de)
- TensorFlow & Keras
- Mistral LLM and Hugging Face
