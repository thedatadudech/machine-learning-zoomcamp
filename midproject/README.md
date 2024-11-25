# Machine Learning Zoomcamp Midterm Project

<img src="assets/project_image.png" alt="Project Image" width="800">

## Disclaimer

This project is intended to fulfill the requirements of the Machine Learning Zoomcamp midterm project. It aims to conduct an analysis without performing a comprehensive qualitative machine learning analysis.

## Project Description

The objective of this project is to predict the heating load of buildings based on various architectural features. The dataset used for this project is sourced from the UCI Machine Learning Repository and contains features such as relative compactness, surface area, wall area, and more.

## Instructions to Run the Project

### Prerequisites

- Python 3.9 or later
- Docker

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script:**

   ```bash
   python train.py
   ```

4. **Run the prediction service:**

   ```bash
   python predict.py
   ```

5. **Interact with the service:**
   Use the `curl.py` script to send a POST request to the prediction service:
   ```bash
   python curl.py
   ```

### Docker Setup

1. **Build the Docker image:**

   ```bash
   docker build -t ml-zoomcamp-project .
   ```

2. **Run the Docker container:**
   ```bash
   docker run -p 5100:5100 ml-zoomcamp-project
   ```

## Data

The dataset used in this project is the "Energy Efficiency" dataset from the UCI Machine Learning Repository. You can download it directly from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx).

## Notebook

A Jupyter notebook (`notebook.ipynb`) is included in the repository, which contains:

- Data preparation and cleaning
- Exploratory Data Analysis (EDA) and feature importance analysis
- Model selection process and parameter tuning

## Scripts

- **train.py**: This script handles the training of the final model and saves it to a file using pickle.
- **predict.py**: This script loads the trained model and serves it via a Flask web service.

## Dependencies

Dependencies are listed in the `requirements.txt` file. You can install them using pip.

## Dockerfile

A `Dockerfile` is provided to run the service in a containerized environment.

## Deployment

The service can be deployed using Docker. Alternatively, you can deploy it on a cloud platform and provide the URL to the service.

## Interaction

You can interact with the deployed service using the `curl.py` script, which demonstrates how to send a request to the prediction endpoint.

---

This project is a simplified demonstration of a machine learning workflow, focusing on fulfilling the course requirements rather than providing a full-fledged analysis.
