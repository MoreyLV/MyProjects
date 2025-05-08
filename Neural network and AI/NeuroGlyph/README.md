# NeuroGlyph

---!!!Remember that you need to install the necessary libraries!!!---

**NeuroGlyph** is a minimal neural network sandbox powered by FastAPI. It allows users to either train or test a simple neural network model directly from the terminal or via a web interface.

-Features-

- Simple fully connected neural network built with NumPy.
- Train or recognize patterns using 3x5 binary input grids.
- Web interface using FastAPI and Jinja2 templates.
- Saves weights and biases in plain `.txt` files.

-Project Structure-

NeuroGlyph/

├── app.py # FastAPI backend with WebSocket support

├── main.py # Command-line interface for training/testing

├── weights.txt # Stored weights of the neural network

├── bias.txt # Stored biases of the neural network

├── templates/

│ └── index.html # Web interface template

├── static/

│ └── fonts/ # Font assets

└── README.md

-How to Run-

---Bash---

python main.py

You'll be prompted to select training or recognition mode.

---Web Interface---

uvicorn app:app --reload

Then visit: http://127.0.0.1:8000

-License-
This project is released under the following conditions:

Not permitted for commercial use.
You may modify, study, and distribute it freely for personal or educational purposes.
Any derivative work must credit the original author and link back to this repository.
Reuploading or reselling this project as-is is not allowed.
