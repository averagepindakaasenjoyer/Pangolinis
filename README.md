# Pangolins Housing Classification Project

This repository contains the code and notebooks developed by the Pangolins team for the University of Amsterdam (UvA). We were selected by **Matrixian** to work on the classification of houses based on provided property data.

## Project Structure

Data/ <br>
│ <br>
├── img_dataset/ <br>
│ ├── 01/ <-- contain .jpg and .json of properties <br>
│ ├── 02/ <br>
│ └── etc... <br>
├── cleaned_sample_with_urls.csv<br>
├── detailed_woning_type_sample.parquet<br>
└── bag_image_summary.csv<br>
<br>
PANGOLINS/ <-- Files in the repository<br>
│<br>
├── Data Analytics/<br>
│ ├── Datavisualisation.ipynb<br>
│ └── Metrics.ipynb<br>
│<br>
├── Model Architecture/<br>
│ ├── chunky_cnn.ipynb<br>
│ └── simple_cnn.ipynb<br>
│<br>
├── Model Pipeline/<br>
│ ├── models/<br>
│ │ └── best_model.pth<br>
│ ├── pipeline_doc_example.py<br>
│ ├── pipeline_notebook.ipynb<br>
│ └── pipelineClass.py<br>
│<br>
├── Preprocessing/<br>
│ ├── Bag.ipynb<br>
│ ├── preprocessing_pipeline.ipynb<br>
│ ├── Preprocessing.ipynb<br>
│ └── Standardize.ipynb<br>
│<br>
├── .env<br>
├── .gitignore<br>
└── README.md<br>
<br>

## Documentation

This section will walk you through the functionalities of the pipeline created.

#### 1. Setup

To make sure all of the files run smoothly, it is important to run in a fresh environment with the correct packages.

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

After, create an `.env` file. In this `.env` file you should have a `FILE_PATH = 'relative/path/to/Data'`, this will be read by the scripts to locate the data. Make sure the `Data/` folder contains the files listed in the structure above.

After making sure the data can be loaded in, execute the `Preprocessing.ipynb`. This will create a file in the `Data/` folder called `Full_preprocessed_detailed_house.csv`. This file will be used by the rest of the models and scripts.

#### 2. Creating a Model

To start using a model, you have two options.

###### 1. Train your own model

To train your own model, you will need to create a `class` that satisfies the following requirements:

- The class has the folowing arguments:
  - **model_class (nn.Module)**: The PyTorch model class to be trained.
  - **csv_path (str)**: Path to the main CSV data file.
  - **image_base_dir (str)**: The base directory where images are stored.
  - **image_col (str)**: The name of the column in the CSV containing the relative image paths.
  - **target_col (str)**: The name of the target variable column.
  - **numeric_cols (list)**: A list of column names for numeric features.
  - **categorical_cols (list)**: A list of column names for categorical features.
  - **epochs (int)**: Number of training epochs.
  - **batch_size (int)**: Batch size for DataLoaders.
  - **lr (float)**: Learning rate for the optimizer.
  - **image_size (tuple)**: The size (height, width) to which images will be resized.
  - **save_filename (str)**: The filename for saving the best model.
  - **device (str, optional)**: The device to run on ('cuda' or 'cpu')- Defaults to auto-detection.
  - **useMask (bool)**: Whether to include mask data in the dataset and model.
- It contains a `forward()` method returning the classifier.

For an example of training have a look at `MaskMultimodalTrain.py`, where an example model has been defined and can be trained.

###### 2. Load a saved model

The second option is to load a model. Take a look at `classify.ipynb` for an example of loading and classifying.

To load a model use the `YourMultimodelClass.load_saved_model('models/your_model.pth', evaluate=False)`. The `evaluate` argument is to specify whether it should load the model in _evaluation_ mode, or if you will be training it further.

#### 3. Using a model

Here is an overview of the available functions once a model and `MultimodelModelClass` have been created:

- `train(epochs(optional))`
  - Runs the full training and validation loop for the multimodal model. It tracks training and validation metrics (loss, accuracy, F1-score) and saves the best performing model based on validation F1-score.
  - **Parameters**: - `epochs (int, optional)`: The number of training epochs to run. If provided, it will override the epochs value set during pipeline initialization. If None, the initialized epochs value will be used.
    <br>
- `evaluate(dataloader_type='test', show_confusion_matrix=True, show_training_curves=True, show_feature_importance=True)`
  - Evaluates the trained model on either the validation or test dataset. This function loads the best-saved model (if available) and calculates various classification metrics such as accuracy, precision, recall, F1-score, Cohen's Kappa, and Log Loss. It can optionally display a confusion matrix, training history curves, and an analysis of tabular feature importance. It is possible to continue training after using `evaluate()`.
  - **Parameters**: - `dataloader_type (str, optional)`: Specifies which dataset to evaluate on. Can be `'test'` (default) or `'val'`.
    - `show_confusion_matrix (bool, optional)`: If `True` (default), a confusion matrix will be plotted.
    - `show_training_curves (bool, optional)`: If `True` (default), plots of training/validation loss, accuracy, and F1-score over epochs will be displayed.
    - `show_feature_importance (bool, optional)`: If `True` (default), an analysis of tabular feature importance using gradient-based methods will be performed and visualized.
      <br>
- `load_saved_model(path, evaluate=True)`
  - Loads a previously saved model's state dictionary into a new instance of the model architecture. This is useful for loading a pre-trained model for inference or further training.
  - **Parameters**:
    - `path` (str): The file path to the saved model's state dictionary (e.g., `'models/best_model.pth'`).
    - `evaluate` (bool, optional): If `True` (default), the loaded model will be set to evaluation mode (`model.eval()`). Set to `False` if you intend to continue training the loaded model.
  - **Returns**:
    - `torch.nn.Module`, The loaded PyTorch model.
      <br>
- `classify(input_csv_name='input.csv', input_image_dir='images', threshold=0.5)`
  - Classifies one or more unlabeled data entries provided in a CSV file, along with their corresponding images. This function preprocesses the input data using the pipeline's fitted scalers and encoders, makes predictions using the loaded model, and returns the predicted class name, probabilities, and label index for each entry. It also supports an "Uncertain" classification based on a specified probability threshold.
  - **Parameters**:
    - `input_csv_name` (str, optional): The filename of the CSV containing the input data. This file is expected to be located in a directory named `'input'` (i.e., `./input/input.csv`). Defaults to `'input.csv'`.
    - `input_image_dir` (str, optional): The subdirectory within the `'input'` folder where the images referenced in the CSV are stored (e.g., `./input/images/`). Defaults to `'images'`.
    - `threshold` (float, optional): A probability threshold. If the maximum predicted probability for an entry is below this threshold, the entry will be classified as `'Uncertain'`. Defaults to `0.5`.
  - **Returns**
    - `List[Tuple[str, List[float], int]]`: A list of tuples, where each tuple represents a prediction for an input entry. Each tuple contains:
      - `str`: The predicted class name or `'Uncertain'`.
      - `List[float]`: A list of probabilities for each class.
      - `int`: The index of the predicted label. Returns an empty list `[]` if an error occurs.

#### 4. Classifying House Types

Finally, to classify house types it is required to place the data in `.csv`-format in the `./input` folder. The input file should contain the following columns:

- `opp_pand` (int)
- `build_year` (int)
- `build_type` (type)
- `geometry_wkt` (str, BAG Polygon)
- `frontview_url` (`./input/images/IMAGE_NAME.jpg`)

In `./input/images/` images should correspond to the `frontview_url` column in the input file.

After, you can use the `classify()` function as documented above. Also look at `classify.ipynb` for a classification example.
