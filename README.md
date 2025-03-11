# Vehicle License Plate Detection & Cropping

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-red)
![Status](https://img.shields.io/badge/Status-Active-success)

## 😎 About the Project
This project detects vehicle license plates from images and applies perspective correction for better readability. It utilizes **YOLOv8** for plate detection and **OpenCV** for image processing.

---

## 📌 Features

✅ License plate detection using YOLOv8 
✅ Automatic plate cropping with padding
✅ Perspective correction using **K-Means Clustering**
✅ Interactive corner selection for better accuracy
✅ Modular and easily expandable codebase

---

## 📂 Folder Structure
```
├── images/          # Input images
├── input/           # Cropped license plates
├── output/          # Perspective-corrected plates
├── best.pt          # YOLOv8 trained weights
├── detect_license_plate.py  # Plate detection script
└── perspective_correction.py  # Perspective correction script
```

---

## ⚙️ Installation
Make sure you have **Python 3.8+** installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Run
### **Step 1: Detect License Plates**
```bash
python detect_license_plate.py
```
This step extracts license plates and saves them in the `input/` folder.

### **Step 2: Apply Perspective Correction**
```bash
python perspective_correction.py
```
A window will appear:
- Adjust **green points** manually.
- Press **Space** to save.
- Press **n** to skip an image.
- Press **q** to quit.

---

## 🖼️ Sample Output
| Original Image | Detected Plate | Corrected Plate |
|---------------|---------------|-----------------|
| ![Original](https://via.placeholder.com/150) | ![Detected](https://via.placeholder.com/150) | ![Corrected](https://via.placeholder.com/150) |

---

## 🚀 Future Improvements
- Train YOLOv8 on a larger dataset for improved accuracy
- Automate corner detection using deep learning
- Extend support for real-time video detection

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

### 📜 License
This project is **MIT licensed**.

---

**👨‍💻 Developed by:  BITA**

