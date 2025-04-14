# Remember to start venv with source env/bin/activate  # macOS/Linux

import pandas as pd  # Import pandas for data manipulation
import time  # Import time to measure how long the backtest takes

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip().capitalize() for col in df.columns]
    df = df[['Date', 'Close']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def buy_and_hold(df):
    entry_price = df['Close'].iloc[0]
    exit_price = df['Close'].iloc[-1]
    return_pct = (exit_price - entry_price) / entry_price * 100
    return entry_price, exit_price, return_pct

def evaluate_performance(entry_price, exit_price, return_pct):
    print(f"Buy at ${entry_price:.2f}")
    print(f"Sell at ${exit_price:.2f}")
    print(f"Total Return: {return_pct:.2f}%")

def run_backtest(file_path, strategy):
    print(f"Starting backtest with {strategy.__name__} strategy...")
    start_time = time.time()

    # Load Data
    df = load_data(file_path)

    # Apply Strategy
    entry_price, exit_price, return_pct = strategy(df)

    # Evaluate Performance
    evaluate_performance(entry_price, exit_price, return_pct)

    # Print Execution Time
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")

# Run backtest with AAPL and the Buy-and-Hold strategy
run_backtest('aapl.csv', buy_and_hold)