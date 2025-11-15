# Dual-Camera Live Stitching (OpenCV + ORB)

This project uses **two webcams** to create a **real-time stitched panoramic output** using ORB feature detection, homography estimation, perspective warping, and overlap blending.

---

## 🚀 Features
- Live dual-camera capture
- ORB feature detection + Hamming matching
- RANSAC-based homography computation
- Real-time perspective warping
- Smooth overlap blending
- Fullscreen layout:
  - **Top:** Stitched view  
  - **Bottom Left:** Left camera  
  - **Bottom Right:** Right camera  
- Save stitched output using `s`
- Recompute homography using `h`
- Quit using `q`

---

## 🛠 Requirements
Install the dependencies:

```bash
pip install opencv-python numpy
```

Hardware needed:
- Two USB webcams
- A system capable of simultaneous capture

---

## ▶️ How to Run

```bash
python stitch.py
```

The program opens a fullscreen display showing stitched and individual camera views.

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **q** | Quit program |
| **s** | Save stitched image |
| **h** | Recompute homography |

---

## 📂 Project Structure

```
.
├── stitch.py
├── stitched_images/
└── README.md
```

---

## 🧩 How It Works

1. Capture frames from left & right cameras  
2. Detect ORB features and compute descriptors  
3. Match descriptors using BFMatcher (Hamming distance)  
4. Use RANSAC to compute homography  
5. Warp right image into left image’s perspective  
6. Find overlap region  
7. Smoothly blend overlapping columns  
8. Display stitched + raw camera views  

---

## 📈 Future Improvements
- Multi-band blending (Laplacian pyramids)
- Auto exposure correction
- Camera calibration to remove lens distortion
- GPU-accelerated ORB (OpenCV CUDA)
- Video recording output

---

## 📝 License
MIT License

---

## 🤝 Contributing
Pull requests are welcome! Improvements to blending, UI layout, or calibration are appreciated.
