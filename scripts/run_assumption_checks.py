import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson

VIF_THRESHOLD = 5.0
SHAPIRO_P_THRESHOLD = 0.05
DW_LOWER = 1.5
DW_UPPER = 2.5

def calculate_vif(x):
    vif_df = pd.DataFrame()
    vif_df["features"] = x.columns
    vif_df["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif_df

def check_linearity(x,y, model):
    os.makedirs("plots",exist_ok=True)
    for features in x.columns:
        plt.figure()
        sns.regplot(x=x[features], y=y, lowess=True, line_kws={'color':'red'})
        plt.title(f"Linearity Check: {features} vs Target")
        filename = f"plots/linearity_{features}.png"
        plt.savefig(filename)
        mlflow.log_artifact(filename, artifact_path="linearity")
        plt.close()

def check_homoscedasticity(x,y,model):
    preds = model.predict(x)
    residuals = y - preds
    plt.figure()
    sns.scatterplot(x=preds, y=residuals)
    plt.axhline(0,color='red',linestyle='--')
    plt.title("Homoscedasticity: Residuals vs Predictions")
    filename = "plots/homoscedasticity.png"
    plt.savefig(filename)
    mlflow.log_artifact(filename, artifact_path="homoscedasticity")
    plt.close()

def check_normality_of_residuals(residuals):
    stat, p = shapiro(residuals)
    mlflow.log_metric("Shapiro_p_value", p)
    return p > SHAPIRO_P_THRESHOLD

def check_autocorrelation(residuals):
    dw_start = durbin_watson(residuals)
    mlflow.log_metric("Durbin Watson", dw_start)

def run_checks(data_path, model_path, target_col):
    df = pd.read_csv(data_path)
    y = df[target_col]
    x = df.drop(columns=[target_col])

    model = joblib.load(model_path)
    preds = model.predict(x)
    residuals = y - preds

    with mlflow.start_run(run_name = "Assumption Checks"):
        mlflow.log_param("model file", model_path)

        check_linearity(x,y,model)
        vif_df = calculate_vif(x)
        vif_df.to_csv("plots/vif_table.csv", index = False)
        mlflow.log_artifact("plots/vif_table.csv", artifact_path="vif")

        check_homoscedasticity(x,y,model)

        if not check_normality_of_residuals(residuals):
            print("Residuals are not normally distributed.")

        if not check_autocorrelation(residuals):
            print("Residuals are correlated")

        print("Assumption Checks Completed")