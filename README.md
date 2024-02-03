# Richter's Predictor: Modeling Earthquake Damage

Predict the level of damage to buildings caused by earthquakes.

![Damage from 2015 Gorkha earthquake in Nepal](https://drivendata-prod-public.s3.amazonaws.com/comp_images/earthquake.jpg)

Based on aspects of building location and construction, this project predicts the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal.

# Demo

Deployed on Streamlit.

# Dataset

This project uses data provided by [DrivenData](https://www.drivendata.org/competitions/57/nepal-earthquake/). The data was collected through surveys by [Kathmandu Living Labs](https://kathmandulivinglabs.org/) and the Central Bureau of Statistics, which works under the National Planning Commission Secretariat of Nepal. This survey is one of the largest post-disaster datasets ever collected, containing valuable information on earthquake impacts, household conditions, and socio-economic-demographic statistics.

## Exploratory Data Analysis
Here we some insights from our exploratory data analysis. 

**Class imbalance of the target, [Link to interactive plot](graphics/class_imbalance.html)**
<img src="graphics/class_imbalance.png" alt="class_imbalance" width="400"/>

**Correlation matrix of numerical features, [Link to interactive plot](graphics/correlation_matrix.html)**
<img src="graphics/correlation_matrix_plot.png" alt="corr_matrix" width="400"/>


# Methods

Our team of data scientists from [Data Science Retreat](https://datascienceretreat.com/) in Berlin developed a model to predict the level of damage to the buildings. We used a variety of machine learning models, including Logistic Regression, Gradient Boosting, XGBoost, and Neural Networks. We also used a variety of feature engineering techniques, including one-hot encoding, target encoding, frequency encoding, and feature scaling.

# Tools

- Python
- Pandas
- Numpy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit

# Results

# Project Structure

```
TBA
```

# Project Setup

## Register with DrivenData and download the data

- Sign up to DrivenData and join the Richter's Predictor competition: [DrivenData](https://www.drivendata.org/competitions/57/nepal-earthquake/)
- Download the data files and save them to the `data` folder: [Download data](https://www.drivendata.org/competitions/57/nepal-earthquake/data/)

## Setup the project

- Navigate to your projects folder, e.g. `cd ~/projects`
- Clone the repository: `git clone https://github.com/carecodeconnect/richters-predictor`
- Navigate to the project folder: `cd richters-predictor`
- Create a virtual environment: `conda create -n richters-predictor python=3.10`
- Activate the virtual environment: `conda activate richters-predictor`
- Install the required packages: `pip install -r requirements.txt`
- Install Jupyter Notebook: `conda install jupyter`
- Install Streamlit: `conda install -c conda-forge streamlit`

## Run the project

- Open the Jupyter Notebook: `jupyter notebook`
- Open and run the notebook: `baseline.ipynb`
- Open and run the best model: `best.py`
- Deploy the app: `streamlit run app.py`
