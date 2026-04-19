# Biceps Curl Classifier (AMAN_AI Equipe)

This project is a real-time (and post-exercise) Biceps Curl Quality Classifier. It uses a 1D CNN trained on Accelerometer and Gyroscope data to distinguish between "Perfect" and "Imperfect" curls.

## Project Structure

- **`android/`**: Android application (Kotlin).
- **`src/`**: Python source code for the model (`model.py`), data utilities (`utils.py`), and training (`train.py`).
- **`data/`**: Directories for `perfect` and `imperfect` sensor samples.
- **`scripts/`**: Helper scripts.
- **`models_saved/`**: Exported TorchScript models for mobile.

## Workflow

### 1. Data Collection
1. Install the Android app on your device.
2. Click **Start Record**, perform a curl, then click **Stop Record**.
3. Label the recording as **P** (Perfect) or **I** (Imperfect).
4. Data is saved to `/sdcard/Android/data/com.example.curl_piseps/files/data`.

### 2. Pulling Data to PC
Use the provided script to retrieve your recordings:
```bash
bash scripts/pull_data.sh
```

### 3. Training the Model
Use Docker to train the model with all dependencies:
```bash
# Build the trainer image
docker build -t curl-trainer .

# Run the training
sudo docker run -v $(pwd):/app curl-trainer python3 src/train.py
```
*The training script will automatically save the best model based on validation loss and print the normalization stats.*

### 4. Data Exploration with Notebooks
To use the Jupyter notebooks in the `notebooks/` folder:
```bash
sudo docker run -p 8888:8888 -v $(pwd):/app curl-trainer jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```
*Access the link provided in the terminal (usually http://localhost:8888) to start experimenting.*

### 5. Deploying the Model
Copy the exported model to the Android assets:
```bash
cp models_saved/curl_classifier.pt android/app/src/main/assets/
```

## Technical Details

- **Sensor Fusion**: Uses both Accelerometer and Gyroscope (6 channels total).
- **Preprocessing**: 4th order Butterworth low-pass filter (10Hz cutoff).
- **Inference**: Analysis is performed on the entire movement buffer after "Stop" is clicked, using an averaged sliding window of 100 samples (2 seconds at 50Hz).
- **Normalization**: Inputs are standardized using Mean/Std calculated during training.

---
**Equipe AMAN_AI**
