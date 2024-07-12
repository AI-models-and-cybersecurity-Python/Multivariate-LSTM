import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    X = df.loc[:, "PM2.5":"RAIN"]
    
    plt.figure()
    plt.plot(X["PM2.5"])
    plt.title("PM2.5 value")
    plt.xlabel("Measurement")
    plt.ylabel("Value")
    plt.show()
    
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    X_train, X_test = train_test_split(X, train_size=0.8, shuffle=False)
    
    return X_train, X_test

def create_generators(X_train, X_test, length=5, batch_size=5):
    train_gen = TimeseriesGenerator(X_train, X_train, length=length, batch_size=batch_size)
    test_gen = TimeseriesGenerator(X_test, X_test, length=length, batch_size=batch_size)
    return train_gen, test_gen

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(input_shape[1]))
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model

def train_model(model, train_gen, epochs=50):
    model.fit(train_gen, epochs=epochs)
    return model

def predict_and_visualize(model, test_gen, X_test):
    preds = model.predict(test_gen)
    
    plt.figure()
    plt.plot(X_test[5:len(preds)+5, 0], label="True Value")
    plt.plot(preds[:, 0], label="Predicted Value")
    plt.xlabel("Measurement")
    plt.ylabel("Value")
    plt.title("Pollution prediction")
    plt.legend()
    plt.show()
    
    return preds

if __name__ == "__main__":

    filepath = "PRSA_Data_Dingling_20130301-20170228.csv"

    X_train, X_test = load_and_preprocess_data(filepath)
    
    train_gen, test_gen = create_generators(X_train, X_test)

    input_shape = (train_gen.length, X_train.shape[1])
    model = build_model(input_shape)

    model = train_model(model, train_gen)

    preds = predict_and_visualize(model, test_gen, X_test)
