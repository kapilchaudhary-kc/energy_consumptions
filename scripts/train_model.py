import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def read_csv():
    path = 'data/energy_consumption.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find the CSV file at: {path}")
    return pd.read_csv(path)

def preprocess(df):
    x = df.drop(columns=['timestamp','energy_consumption'])
    y = df['energy_consumption']
    return x,y

def train(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=21)
    
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(x_train,y_train)

        preds = model.predict(x_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        joblib.dump(model,'models/lin_reg.pkl')
        mlflow.log_artifact('models/lin_reg.pkl',artifact_path='model')
    return x_test,y_test

def test_data(x_test,y_test):
    test_data = x_test.copy()
    test_data['energy_consumption'] = y_test
    test_data.to_csv(r'data\processed_test_data.csv',index=False)
    print("Model Trained and Saved. Test data saved for the assumption checks.")

def main():
    df = read_csv()
    x,y = preprocess(df)
    x_test, y_test = train(x,y)
    test_data(x_test, y_test)

if __name__ == "__main__":
    main()
