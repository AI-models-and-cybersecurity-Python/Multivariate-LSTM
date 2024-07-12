# Multivariate LSTM

This project leverages TensorFlow and Keras libraries to construct and train a Multivariate LSTM model. The primary objective of the repository is to predict pollution levels using the dataset `PRSA_Data_Dingling_20130301-20170228.csv`.

## Project Structure

- `MultivariateLSTM.py`: The main script that includes functions for loading data, preprocessing, building the LSTM model, training the model, and evaluating the model.
- `PRSA_Data_Dingling_20130301-20170228.csv`: The dataset used for training and testing the model.
- `requirements.txt`: A list of Python libraries required to run the script.

## Installation

To get started with this project, follow these steps:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/AI-models-and-cybersecurity-Python/Multivariate-LSTM.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Multivariate-LSTM
    ```

3. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure that the dataset `PRSA_Data_Dingling_20130301-20170228.csv` is placed in the project directory.

2. Run the script:
    ```bash
    python MultivariateLSTM.py
    ```

## Results

After training, the script will display a plot showing the true and predicted pollution levels. This allows for a quick visual check of the model's performance.

## License

This project is licensed under the MIT License. 
