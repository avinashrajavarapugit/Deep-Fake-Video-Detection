
# 🎥 Deepfake Detection System  

This project leverages state-of-the-art technologies to detect **deepfakes** by analyzing audio and video synchronization using advanced machine learning models.

---
## Site Link: 
- https://deepfakevideodetection-gvla.onrender.com [**Currently Backend dowm due to lack of enough RAM**]
---
## 🧑‍💻 Key Features  

- **Real-Time Deepfake Detection**: Analyze videos and determine if they are deepfakes.  
- **Audio-Visual Synchronization**: Measure audio-visual alignment for deepfake classification.  
- **Lightweight Dockerized Setup**: Deploy anywhere using Docker.  
- **End-to-End Automation**: Includes preprocessing, feature extraction, and inference.  

---

## 🛠️ Tech Stack  

This project is built using the following technologies:  

- **Programming Languages & Frameworks**:  
  - `Python` (Core logic and ML models)  
  - `Flask` (Backend API)  
- **Libraries**:  
  - `NumPy` (Scientific computing)  
  - `TensorFlow` (Deep learning models)  
  - `MoviePy` (Video processing)  
  - `Mediapipe` (Face and landmark detection)  
  - `OpenCV` (Image processing)  
  - `SpeechPy` (Audio feature extraction)  
  - `SciPy` (Signal processing)  
- **Deployment & Containerization**:  
  - `Docker`  

---


## 🎯 How It Works  

1. **Video Preprocessing**  
   - Extract frames using OpenCV.  
   - Detect mouth regions with Mediapipe FaceMesh.  

2. **Audio Preprocessing**  
   - Convert video audio to WAV format.  
   - Extract MFCC features using SpeechPy.  

3. **Model Predictions**  
   - Predict audio and video features using pre-trained SyncNet models (`audio_model` and `lip_model`).  
   - Compute Euclidean distance to measure synchronization.  

4. **Deepfake Detection**  
   - Classify video as a deepfake if the synchronization confidence score exceeds a threshold.  


## 📦 Installation  

### Prerequisites  

- [Docker](https://www.docker.com/get-started) installed on your machine.  

### Steps  

1. **Clone the repository**  
   ```bash  
   git clone https://github.com/avinashrajavarapugit/Deep-Fake-Video-Detection.git  
   ```  

2. **Build the Docker Image**  
   ```bash  
   docker build -t deepfake-recognition-app .  
   ```  

3. **Run the Container**  
   ```bash  
   docker run -p 5000:5000 deepfake-recognition-app  
   ```  

4. **Access the App**  
   Open your browser and navigate to `http://127.0.0.1:5000/`.  

---

## 📸 Sample Output  

![image](https://github.com/user-attachments/assets/0c7f1ed7-bf58-401d-9668-e0d7616d75e0)

- Added an original video

![image](https://github.com/user-attachments/assets/4127e5a8-04bb-482e-86fb-84d9803d4d9b)

- Result
![image](https://github.com/user-attachments/assets/635734ee-ceac-46aa-b657-a00f9ce8db2f)

---

## 🤝 Contributing  

Contributions are welcome! Feel free to open issues, fork the repo, and submit pull requests.  

