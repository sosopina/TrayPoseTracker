# TrayPoseTracker

## Setup Instructions

### Prerequisites
- Python 3.10.17
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/sosopina/TrayPoseTracker.git
cd TrayPoseTracker
```

2. First, install uv if you haven't already:
```bash
pip install uv
```

3. Create a virtual environment with Python 3.10.17:
```bash
uv venv --python 3.10.17
```

4. Activate the virtual environment:
```bash
# On Windows:
.venv/Scripts/activate
# On Unix/MacOS:
source .venv/bin/activate
```

5. Create a `pyproject.toml` file in your project root with this content:
```toml
[project]
name = "TrayPoseTracker"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "av>=14.3.0",
    "carla>=0.9.15",
    "filterpy>=1.4.5",
    "matplotlib>=3.10.1",
    "numpy<2.0",
    "open3d>=0.19.0",
    "opencv-python>=4.11.0.86",
    "pandas>=2.2.3",
    "plotly>=6.0.1",
    "pybind11==2.11.0",
    "pybind11-global==2.11.0",
    "pygame>=2.6.1",
    "pynput>=1.8.1",
    "pyorbbecsdk",
    "rerun-sdk>=0.23.1",
    "scipy>=1.15.2",
    "tqdm>=4.67.1",
    "wandb>=0.19.11",
    "wheel>=0.45.1",
]

[tool.uv.sources]
pyorbbecsdk = { url = "https://github.com/orbbec/pyorbbecsdk/releases/download/v2.0.10/pyorbbecsdk-2.0.10-cp310-cp310-win_amd64.whl" }
```

6. Install all dependencies and run the project:
```bash
uv run --active color.py
```

This command will:
- Install all required dependencies automatically
- Download the Orbbec SDK from the specified wheel file
- Run the color.py script to test the camera

### Testing the Orbbec Camera

The `uv run --active color.py` command will display the camera feed from your Orbbec camera if everything is set up correctly.

### Project Dependencies

The project uses the following main dependencies:
- av>=14.3.0
- carla>=0.9.15
- filterpy>=1.4.5
- matplotlib>=3.10.1
- numpy<2.0
- open3d>=0.19.0
- opencv-python>=4.11.0.86
- pandas>=2.2.3
- plotly>=6.0.1
- pybind11==2.11.0
- pybind11-global==2.11.0
- pygame>=2.6.1
- pynput>=1.8.1
- pyorbbecsdk
- rerun-sdk>=0.23.1
- scipy>=1.15.2
- tqdm>=4.67.1
- wandb>=0.19.11
- wheel>=0.45.1

Note: The pyorbbecsdk is installed from a specific wheel file for Windows Python 3.10.

### Development

To contribute to this project:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

### License
This project is licensed under the terms of the license included in the LICENSE file.
