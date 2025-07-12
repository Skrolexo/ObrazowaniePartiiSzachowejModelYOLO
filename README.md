# Chess Game Imaging with YOLO

A Python-based application for real-time chess game tracking using a webcam and YOLO object detection. Enables gameplay between two human players or a human vs. the Stockfish chess engine.

---

## 🖥 System Requirements

- **Operating System**: Windows 10 or 11  
- **Python**: Version 3.8 or newer  
- **Webcam**: A connected webcam (recommended: use a smartphone as a camera)  

### Optional (for GPU acceleration with CUDA):

- NVIDIA GPU with CUDA support  
- Installed NVIDIA drivers and CUDA Toolkit  
- PyTorch installed with CUDA support  

---

## 🚀 How to Run

> All commands should be executed from the `ObrazowaniePartiiSzachowej` directory

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install the required dependencies

```bash
pip install -r requirements.txt
pip install opencv-python numpy torch ultralytics python-chess
```

### 3. [Optional] Enable GPU support with CUDA

> The default `pip install torch` installs the CPU version. To use CUDA:

```bash
pip uninstall torch
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Make sure your webcam is connected

### 5. Run the program

```bash
python main.py
```

### 6. Select your webcam device

### 7. Choose a game mode:

- **Human vs Human** – for two-player games  
- **Human vs Bot** – play against Stockfish (requires Stockfish engine installed)

### 8. Position your camera about 30 cm above the chessboard  
> Ensure the board fits within the detection grid.

### 9. Move pieces by hand – the system will detect moves after your hand leaves the board view.

---

## ♟ Stockfish Engine

To enable the "Human vs Bot" mode, make sure to download and include the [Stockfish](https://stockfishchess.org/download/) chess engine in the `stockfish/` directory.

---

## 📁 Project Structure

```
ObrazowaniePartiiSzachowej/
├── main.py              # Main application script
├── myChessV9.py         # YOLO model logic for chess piece detection
├── stockfish/           # Stockfish chess engine files
├── requirements.txt     # Python dependency list
```

---

## 📄 License

This project uses the [Stockfish](https://stockfishchess.org/) engine, which is released under the [GNU General Public License v3 (GPLv3)](https://www.gnu.org/licenses/gpl-3.0.html).  
Accordingly, this entire project is distributed under the same license.  
See the included `LICENSE` file for more details.

---

## 👤 Author

Rafał Skolimowski
