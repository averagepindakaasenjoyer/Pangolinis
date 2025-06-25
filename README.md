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
PANGOLINS/ <-- Files in te repository<br>
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

To get started with using the final model you will need to ...

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

After create an `.env` file. In this `.env` file you should have a `FILE_PATH = 'relative/path/to/Data'`, this will be read by the scripts to locate the data. Make sure the `Data/` folder contains the files listed in the structure above.

After making sure the data can be loaded in, execute the `Preprocessing.ipynb`. This will create a file in the `Data/` folder called `Full_preprocessed_detailed_house.csv`. This file will be used by the rest of the models and scripts.

# TODO TODO TODO

Next up load the the saved model from the `best_model.pth`.
