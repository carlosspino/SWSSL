Overview
This class 'train_network_dbt.py' encapsulates a comprehensive framework for training anomaly detection models on high-resolution medical images, specifically focusing on chest or Digital Breast Tomosynthesis (DBT) images. It introduces methods such as 'twin_loss' for calculating loss in a dual network architecture, 'train' for orchestrating the training process, and auxiliary methods like 'create_dataloader' and 'evaluate_and_save_model' for dataset setup and model evaluation. Command-line argument parsing is facilitated through the 'get_arg' method, there are default values if you do not want to especify them.
This class provides a modular and extensible implementation for trainining anomay detection models for medical images apps.


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


During training, the script prints informative updates, including the current triplet loss and other relevant information.

Execution

For the execution of the code you need to execute run_chest.ps1. It is done with the command '.\run_chest.ps1' in the directory where you have this class. 

Notes
The script utilizes GPU acceleration if a CUDA-compatible GPU is available. If not, we will use a CPU
Make sure to modify the dataset path, category, and other parameters as needed for your specific dataset and experiment.
This is only a re-implementation, you are up to change it following the creators schema, with their public github repository.

We have modified the evaluate.py class, adding a portion of code for creating a json with the results because of the troubles we had running the experiments. Also the run_chest class, executing it with bash did not work for us.
We have changed the init and the get_item methods in dataset.py

Original work by Haoyu Dong, Yifan Zhang, Hanxue Gu, Nicholas Konz, Yixin Zhang, and Maciej Mazurowski. Re-implemented by Carlos Pino Padilla and Carlos Ramírez Rodríguez de Sepúlveda.