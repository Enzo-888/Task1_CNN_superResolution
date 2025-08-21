# CNN-based Super-Resolution for Ocean Current Data

This project aims to demonstrate how to use a basic Convolutional Neural Network (CNN) for Super-Resolution (SR) reconstruction of ocean current data (like `u_bar` and `v_bar`). The project includes a complete end-to-end example, a practice template, and data processing scripts.

## Features

- **End-to-End Workflow**: Provides a complete workflow from data loading, model training, validation, to testing and visualization in a Jupyter Notebook ([`Task1_CNN_SR_example.ipynb`](Task1_CNN_SR_example.ipynb)).
- **Basic CNN Model**: Implements a simple yet effective CNN model for the super-resolution task.
- **Result Visualization**: Generates comparison plots to visually showcase the differences between Low-Resolution input (LQ), High-Resolution ground truth (GT), and the model's Super-Resolution output (SR).
- **Model & Prediction Saving**: Automatically saves the best-performing model based on the validation set and can save partial predictions as `.npy` files for further analysis.
- **Hands-on Practice**: Includes a practice notebook ([`Task1_CNN_SR_practice.ipynb`](Task1_CNN_SR_practice.ipynb)) to encourage users to replicate the entire process on the `v_bar` dataset.
- **Data Utility**: Contains a helper script ([`data_demo.py`](data_demo.py)) to create smaller data subsets from the full dataset, facilitating quick debugging and experiments.

## Project Structure

```
.
├── data/
│   ├── u_bar/              # U-component ocean current data
│   │   ├── ubar_hr_train.npy
│   │   ├── ubar_sr_input_train.npy
│   │   └── ...
│   └── v_bar/              # V-component ocean current data
│       ├── vbar_hr_train.npy
│       └── ...
├── sr_cnn_results_ubar_demo/ # Output directory for the example code
│   ├── best_model.pth
│   ├── test_visualization.png
│   └── validation_npy_predictions/
│       └── ...
├── data_demo.py            # Script for creating a small-scale demo dataset
├── Task1_CNN_SuperResolution_example.ipynb # Core example: Complete SR workflow using u_bar data
├── Task1_CNN_SuperResolution_practice.ipynb # Practice task: Replicate the SR workflow
├── README.md               # This document
└── READ.txt
```

## Requirements

Please ensure you have the following Python libraries installed. You can install them using pip:

```sh
pip install torch numpy matplotlib
```

## Usage

### 1. Run the Core Example

Open and run [`Task1_CNN_SR_example.ipynb`](Task1_CNN_SR_example.ipynb) directly in a Jupyter environment.

- This notebook will use the data from the `data/u_bar` directory.
- It will perform model training, validation, and testing.
- All outputs (best model, visualization images, predicted .npy files) will be saved in the `sr_cnn_results_ubar_demo/` directory.

### 2. Complete the Practice Task

Open [`Task1_CNN_SR_practice.ipynb`](Task1_CNN_SR_practice.ipynb) and, following the prompts and referencing the example code, complete the code to build a new super-resolution task for the `v_bar` dataset.

### 3. Create a Demo Dataset (Optional)

If you want to use a smaller dataset for quick testing, you can run [`data_demo.py`](data_demo.py).

1.  Open the [`data_demo.py`](data_demo.py) file.
2.  Modify the `DEMO_SAMPLE_COUNT` variable to define the number of samples for each dataset.
3.  Modify the file paths in `file_processing_list` to ensure they point to your original data and desired output locations.
4.  Run the script:
    ```sh
    python data_demo.py
    ```

## Output

After running [`Task1_CNN_SR_example.ipynb`](Task1_CNN_SR_example.ipynb), the following will be generated in the [`sr_cnn_results_ubar_demo/`](sr_cnn_results_ubar_demo) directory:

- `best_model.pth`: The model weights file with the lowest loss on the validation set.
- `test_visualization.png`: A comparison plot with 5 subplots showing LQ, GT, SR, and the errors between them.
- `validation_npy_predictions/`: A subdirectory containing several `.npy` files, which save the LQ, GT, and SR data for some validation samples.

## Getting Started

1.  **Clone/Download the Project**: Download this project to your local machine.
2.  **Prepare the Data**: Ensure the `data/` directory contains the `u_bar` and `v_bar` data. If the data is missing, you may need to run a data preparation script (if provided) or place it manually.
3.  **Install Dependencies**: `pip install torch numpy matplotlib`.
4.  **Run the Example**: Open and run [`Task1_CNN_SR_example.ipynb`](Task1_CNN_SR_example.ipynb) from start to finish to understand the entire workflow and see the results.
5.  **Hands-on Practice**: Try to complete the exercises in [`Task1_CNN_SR_practice.ipynb`](Task1_CNN_SR_practice.ipynb) to solidify your understanding.

## Notes

- **Device Configuration**: The code will automatically detect and prioritize using a CUDA GPU. If no GPU is available on your machine, it will fall back to CPU execution. Training on a CPU can be very time-consuming.
- **Path Configuration**: All paths in the notebooks and scripts are relative. Please ensure you run them from the project's root directory, or you may need to adjust the file paths.
- **Data Format**: This project uses the `.npy` file format for data storage. The data is assumed to be a 3D array of shape `(num_samples, height, width)`, where `height` and `width` are the dimensions of the ocean current data fields.
