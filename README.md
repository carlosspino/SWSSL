Overview
This script is designed for training an anomaly detection model using a Siamese network with triplet loss. The model is trained on chest images, and the training process involves learning features that can discriminate between positive (normal) and negative (anomalous) samples.

Requirements
Before running the script, ensure you have the following dependencies installed:

Python 3.6 or higher
PyTorch
torchvision
scikit-learn
tqdm
OpenCV
NumPy
PIL (Pillow)
Install the required Python packages using the following command:

bash
Copy code
pip install torch torchvision scikit-learn tqdm opencv-python numpy pillow
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your_username/your_repository.git
cd your_repository
Run the training script:
bash
Copy code
python train_anomaly_detection.py
The script supports various command-line arguments, including:

--phase: Choose between training and testing phases.
--dataset_path: Path to the dataset directory (default: '../dbt_dataset').
--category: Choose between 'chest' and 'dbt' categories (default: 'dbt').
--batch_size: Batch size for training (default: 300).
--lr: Learning rate for the optimizer (default: 1e-4).
--epochs: Number of training epochs (default: 10000).
--patch_size: Size of image patches for training (default: 128).
--step_size: Step size for extracting patches (default: 32).
Adjust these parameters according to your specific requirements.

Model and Training Process
The script uses a Siamese network with triplet loss for training. The training process involves optimizing the model's parameters to learn discriminative features. The triplet loss consists of positive and negative pairs, encouraging the model to pull positive samples closer together in feature space and push negative samples apart.

During training, the script prints informative updates, including the current triplet loss and other relevant information.

Notes
The script utilizes GPU acceleration if a CUDA-compatible GPU is available.
Make sure to modify the dataset path, category, and other parameters as needed for your specific dataset and experiment.
Feel free to explore and customize the script based on your specific use case and dataset characteristics. For further assistance, please refer to the script's source code or contact the repository owner.
