# 🎗️ CerviScan Website

**CerviScan Website** is a web application integrated with a machine learning model designed to detect cervical cancer through IVA (Visual Inspection with Acetic Acid) images. This platform provides tools for patient data input, image uploads for detection, and a history feature to review past analyses.

---

## ✨ **Features**
1. **Patient Data Entry** 📝  
   Easily input and manage patient information, ensuring accurate record-keeping.

2. **IVA Image Upload & Detection** 📸  
   Upload IVA images for real-time detection and analysis powered by advanced machine learning.

3. **History Tracking** 📂  
   View a detailed history of previous detections, including processed images and predictions.

---

## 🧠 **Machine Learning Details**
The detection process is driven by a robust and finely-tuned machine learning pipeline:  

1. **Segmentation**  
   - Utilizes the **Multi-Otsu Thresholding** method to segment regions of interest in IVA images.

2. **Feature Extraction**  
   - Extracts critical image features using techniques such as:  
     - **YUV Color Moments**  
     - **Local Binary Patterns (LBP)**  
     - **Tamura Features**  
     - **Gray Level Run Length Matrix (GLRLM)**  

3. **Classification**  
   - Employs the **XGBoost** classifier, optimized through hyperparameter tuning, to deliver precise predictions.

---

## 🚀 **How It Works**
1. Register or log in to access the application.  
2. Input patient information and upload IVA images.  
3. The system processes the image using the ML pipeline, generating:  
   - Grayscale conversion  
   - Segmented mask  
   - Feature extraction  
   - Prediction (Normal/Abnormal)  
4. View the results and access a comprehensive history of all previous detections.

---

## 🛠️ **Technology Stack**
- **Frontend**: HTML, CSS, Bootstrap  
- **Backend**: Flask (Python)  
- **Database**: SQLite  
- **Image Processing**: OpenCV, Matplotlib  
- **Machine Learning**: XGBoost  

---

## 📚 **Getting Started**
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/cerviscan-website.git
   ```
2. Navigate to the project directory:  
   ```bash
   cd cerviscan-website
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:  
   ```bash
   python app.py
   ```
5. Open your browser and visit `http://localhost:5000`.
