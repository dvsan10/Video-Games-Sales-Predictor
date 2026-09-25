# Video Game Sales Predictor

## Overview

### Problem Statement

Predict the **total sales of a video game** using historical game information such as critic score, console, genre, publisher, developer, and title characteristics.

The project demonstrates an end-to-end machine learning workflow for preparing historical video game data, comparing regression models, and deploying the selected model as a REST API.

### Project Type

- [x] Supervised Learning - Regression
- [ ] Supervised Learning - Classification
- [ ] Unsupervised Learning
- [ ] Time Series Forecasting
- [ ] Recommendation System

### Objective

The primary objective is to develop a regression model that predicts the `total_sales` of a video game using relevant game information while avoiding target leakage from regional sales columns.

---

## Dataset Information

### Dataset Source

The project uses the provided `vgdataset.csv` dataset.

**Dataset Source URL:** The original public dataset URL is not documented in the current project files.

### Dataset Description

The dataset contains historical video game information including title, console, genre, publisher, developer, critic score, regional sales, total sales, release date, and last update information.

Regional sales columns such as `na_sales`, `jp_sales`, `pal_sales`, and `other_sales` are components of `total_sales` and were excluded from the model predictors to prevent target leakage.

### Dataset Size

| Attribute | Value |
|------------|--------|
| Records | 64,016 |
| Original Columns | 13 |
| Target Variable | `total_sales` |
| Target Type | Continuous |
| Final Selected Features | 15 |

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Performed:

- Dataset Overview
- Missing Value Analysis
- Summary Statistics
- Correlation Analysis
- Feature Distribution Analysis
- Duplicate Analysis
- Visualizations

The project includes more than the required 8 visualizations covering distributions, relationships, correlations, and other dataset characteristics.

### Key Insights

- The dataset contains substantial missing values in several sales and metadata columns.
- `total_sales` contains a large number of missing values, so records without a target value were excluded from supervised model training.
- Regional sales columns are components of `total_sales` and were therefore excluded from model predictors to prevent target leakage.
- Video game sales are highly variable, with a relatively small number of games accounting for much larger sales values.
- Publisher and developer performance can provide useful historical information for sales prediction.

---

### 2. Data Preprocessing

Performed:

- Missing Value Analysis
- Missing Value Handling
- Duplicate Removal
- Outlier Analysis
- Date Conversion
- Categorical Encoding
- Frequency Encoding
- Feature Scaling
- Chronological Train-Test Split

Missing `critic_score` values were handled using the median calculated from the training data.

Categorical missing values were represented as:

```text
Unknown
```

Historical publisher and developer features with no previous information were represented as:

```text
0
```

Outlier analysis was performed during EDA. The final modeling workflow did not apply blanket clipping to the target because extreme sales values are meaningful observations in this dataset.

---

### 3. Feature Engineering & Selection

Performed:

- One-Hot Encoding
- Console Frequency Encoding
- Title Feature Creation
- Historical Publisher Feature Creation
- Historical Developer Feature Creation
- SelectKBest Feature Selection
- Standard Scaling

### Feature Creation

The following title-based features were created:

- `title_length`
- `title_word_count`

Historical publisher features:

- `publisher_avg_sales`
- `publisher_previous_games`

Historical developer features:

- `developer_avg_sales`
- `developer_previous_games`

The `genre` feature was encoded using `OneHotEncoder`.

The `console` feature was converted into a frequency-based numerical feature using frequencies calculated from the training data.

### Feature Selection

The top **15 features** were selected using:

```python
SelectKBest(score_func=f_regression, k=15)
```

Feature selection was fitted using the training data and then applied to the unseen test data.

### Final Features Used

The final model uses the selected 15 features produced by `SelectKBest` from the following engineered feature groups:

- Critic score
- Publisher historical features
- Developer historical features
- Title length
- Title word count
- Console frequency
- One-hot encoded genre features

---

### 4. Model Building

Models Implemented:

1. Dummy Regressor
2. Linear Regression
3. Polynomial Regression
4. Ridge Regression
5. Lasso Regression

Polynomial Regression used:

```text
Polynomial Degree = 2
```

A chronological train-test split was used so that earlier observations were used for training and later observations were reserved for testing.

---

### 5. Model Evaluation

#### Evaluation Metrics

| Model | MAE | MSE | RMSE | R² Score |
|---------|---------:|---------:|---------:|---------:|
| Dummy Regressor | — | — | 1.0663 | -0.0058 |
| Linear Regression | — | — | 0.9979 | 0.1191 |
| Ridge Regression | — | — | 0.9979 | 0.1191 |
| Lasso Regression | — | — | 1.0001 | 0.1153 |
| Polynomial Regression | 0.3057 | 0.8743 | 0.9350 | 0.2266 |

### Best Model

Selected Model:

- **Polynomial Regression**

Reason for Selection:

- Lowest test RMSE among the evaluated models.
- Highest test R² score among the evaluated models.
- Lowest reported MAE among the evaluated models.
- Lowest reported MSE among the evaluated models.
- Captures nonlinear relationships between the selected features and total sales.

### Final Model Performance

```text
MAE  = 0.3057
MSE  = 0.8743
RMSE = 0.9350
R²   = 0.2266
```

The R² score indicates that the selected features explain approximately 22.7% of the variation in total video game sales. Other factors not available in the dataset may also influence sales.

---

## Deployment

### Framework Used

- FastAPI
- Uvicorn
- Docker

### API Endpoint

#### Health Check

```http
GET /health
```

#### Prediction

```http
POST /predict
```

### Sample Request

```json
{
    "title": "Grand Theft Auto V",
    "console": "PS3",
    "genre": "Action",
    "publisher": "Rockstar Games",
    "developer": "Rockstar North",
    "critic_score": 9.7
}
```

### Sample Response

```json
{
    "predicted_total_sales": 1.5427
}
```

The prediction value is generated by the trained Polynomial Regression model and may vary depending on the supplied input features.

---

## Docker Containerization

### Build Docker Image

```bash
docker build -t video-game-sales-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 video-game-sales-api
```

The API is then available at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Installation & Setup

### Clone Repository

```bash
git clone https://github.com/dvsan10/Video-Games-Sales-Predictor.git
cd Video-Games-Sales-Predictor
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r deployment/requirements.txt
```

### Run Application

Navigate to the deployment folder:

```bash
cd deployment
```

Run FastAPI:

```bash
uvicorn app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Project Structure

```text
Video-Games-Sales-Predictor/
│
├── dataset/
│   └── vgdataset.csv
│
├── deployment/
│   ├── app.py
│   ├── video_game_sales_model.joblib
│   ├── requirements.txt
│   └── Dockerfile
│
├── notebooks/
│   └── Video Games Sales.ipynb
│
├── screenshots/
│   ├── 01_Docker_Build.png
│   ├── 02_API_Health.png
│   ├── 03_API_Prediction.png
│   └── 04_API_Prediction_Result.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File | Description |
|------|-------------|
| `vgdataset.csv` | Video game sales dataset |
| `Video Games Sales.ipynb` | EDA, preprocessing, feature engineering, model training, and evaluation |
| `app.py` | FastAPI backend and prediction API |
| `video_game_sales_model.joblib` | Saved Polynomial Regression model and preprocessing artifacts |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker configuration for deployment |
| `README.md` | Project documentation |
| `.gitignore` | Files excluded from Git |
| `screenshots/` | Deployment and API screenshots |

---

## Prediction Workflow

The prediction workflow is:

```text
Game Information
       ↓
Feature Creation
       ↓
Genre Encoding
       ↓
Console Frequency Encoding
       ↓
Historical Publisher/Developer Features
       ↓
Feature Selection
       ↓
Feature Scaling
       ↓
Polynomial Transformation
       ↓
Polynomial Regression
       ↓
Inverse Log Transformation
       ↓
Predicted Total Sales
```

---

## Screenshots

### Docker Build

![Docker Build](screenshots/01_Docker_Build.png)

### API Health Check

![API Health](screenshots/02_API_Health.png)

### API Prediction

![API Prediction](screenshots/03_API_Prediction.png)

### API Prediction Result

![API Prediction Result](screenshots/04_API_Prediction_Result.png)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* FastAPI
* Uvicorn
* Joblib
* REST API
* Docker
* Jupyter Notebook
* Machine Learning

---

## Results

- Developed an end-to-end video game sales prediction system.
- Performed EDA, preprocessing, feature engineering, feature selection, and model comparison.
- Evaluated Linear, Polynomial, Ridge, and Lasso Regression models along with a Dummy Regressor baseline.
- Polynomial Regression achieved a test RMSE of **0.9350** and R² of **0.2266**.
- Saved the trained model and preprocessing artifacts using `joblib`.
- Successfully integrated the model with FastAPI.
- Successfully containerized the FastAPI application using Docker.
- Successfully tested the `/health` and `/predict` endpoints through FastAPI Swagger.
- The `/predict` endpoint returned a **200 OK** response during deployment testing.

---

## Key Learning Outcomes

Through this project, I worked on:

* Exploratory Data Analysis
* Missing-value analysis and handling
* Duplicate detection and removal
* Data preprocessing
* Feature engineering
* Categorical encoding
* Frequency encoding
* Historical feature engineering
* Feature selection using SelectKBest
* Feature scaling
* Chronological train-test splitting
* Regression algorithms
* Linear Regression
* Polynomial Regression
* Ridge Regression
* Lasso Regression
* Model comparison
* MAE, MSE, RMSE and R² evaluation
* Overfitting and underfitting analysis
* Target transformation
* Saving and loading ML models
* REST API development with FastAPI
* Docker containerization
* Deploying a machine learning model as an API

---

## Future Improvements

- Improve model performance using additional relevant features.
- Include additional game metadata such as marketing and budget information.
- Add franchise-level information when available.
- Incorporate user ratings and review data.
- Experiment with additional regression algorithms.
- Perform additional hyperparameter tuning.
- Improve historical feature engineering.
- Add automated API testing.
- Deploy the FastAPI backend to a cloud platform.
- Create an interactive frontend for predictions.
- Add model monitoring.
- Implement CI/CD.

---

## Author

**Student Name:** DHARUN VIKASH. R

**Batch:** DS-ANB-03

**Submission Date:** 26-09-2026
