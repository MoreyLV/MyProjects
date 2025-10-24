# 🧩 FormatConverter

**FormatConverter** is a simple graphical application built with **Python (PyQt)** that allows users to convert image files between various data formats.  
It was designed to simplify format and resolution transformation without the need for external tools or command-line usage.

---

## 🚀 Features
- User-friendly Qt-based interface  
- Supports multiple file formats (e.g., `.jpg`, `.jpeg`, `.ico`, `.pdf`)  
- Input/output format selection  
- Saving of converted files
- Manual resolution for image
- Preview of selected image

---

## 🧠 Technologies
- **Python 3.10+**  
- **PyQt5 / PySide6** — for GUI
- **Pillow -for convertation
- **pathib for path interactions
- **PyInstaller** — for `.exe` build  

---

## ⚙️ Installation & Usage

1. Download exe or all .py files
     - if you downloaded exe file, just launch it
     - if you downloaded .py files, make sure both of them are in the same directory
         open main.py via PyCharm and install dependencies by pip install Pillow, pathlib, PySide6
         launch by Ctrl + f5 the main.py file
         

## 📦 Project Structure
```
FormatConverter/
│
├─ main.py              # Entry point and backend
├─ FormatDesign.py      # Qt interface logic
├─ FormatConverter.ui   # GUI layout (Qt Designer)
└─ FormatConverter.exe  # Pre-builded, ready for use by double-click
```

---

## 👨‍💻 Author
**MoreyLV**  
📂 [GitHub: MoreyLV](https://github.com/MoreyLV)

---

## 📝 License
This project is distributed under the **MIT License** — feel free to use and modify it.
