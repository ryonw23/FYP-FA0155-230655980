# AI Powerlifting Buddy

Analyse one prerecorded Squat, Bench Press or Conventional Deadlift repetition. The app uses pose estimation and deterministic measurements to produce feedback. A local Llama model helps word the feedback. Deterministic feedback remains available if local generation is unavailable or rejected.

## Installation

1. Download the project as a ZIP file and extract it, or clone the repository.
2. Install Python 3.12 using the installer for your operating system from the [official Python 3.12.10 release page](https://www.python.org/downloads/release/python-31210/).
3. Open a terminal in the project folder containing `requirements.txt`.
4. Follow the commands for your operating system below.

An internet connection is required for installation and model setup.

### macOS

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m powerlifting_coach.setup
```

### Windows (Command Prompt)

```bat
py -3.12 --version
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m powerlifting_coach.setup
```

The setup command downloads the required MediaPipe and Llama models into `models/`. Running it again reuses valid model files.

## Launch

Complete the installation steps above before launching the app. You do not need to repeat installation or model setup each time.

1. Open a terminal in the project folder containing `requirements.txt`.
2. Activate the virtual environment and start the app using the commands for your operating system.

### macOS

```bash
source .venv/bin/activate
python -m powerlifting_coach.app
```

### Windows (Command Prompt)

```bat
.venv\Scripts\activate.bat
python -m powerlifting_coach.app
```

3. Wait for a local URL to appear in the terminal, then open that URL in your web browser.
4. Select Squat, Bench Press or Conventional Deadlift. Upload a video containing one visible lifter and one complete repetition, then press **Analyse**.

Keep the terminal open while using the app. To stop it, return to the terminal and press **Ctrl+C**. After installation and model setup, the app can run offline.

The app provides recording guidance for each lift. The suggested views came from a visual comparison of tracking across the tested recordings and do not establish biomechanical accuracy.

Generated annotated videos and JSON results are saved under `output/`.
