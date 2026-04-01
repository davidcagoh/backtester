# backtester

**backtester** is a lightweight web app for turning a plain-English trading idea into executable strategy logic and immediate historical backtest results. The core interaction is simple: describe a strategy in natural language, let the model translate it into Python, and run it against local CSV price data without leaving the browser.

The repository is best read as a compact product prototype rather than a finance library. Its value lies in the full loop from **natural-language strategy specification** to **code generation**, **simulation**, and **interactive result presentation**.

![Backtester interface](img1.png)
![Backtester results view](img2.png)

| Repository focus | Description |
|---|---|
| Product idea | Plain-English strategy creation with immediate backtesting |
| Input format | Local CSV time-series data supplied by the user |
| Core output | Generated strategy code, performance metrics, equity curve, and trade log |
| Technical stack | Python, pandas, browser UI, LLM-assisted code generation |
| Portfolio value | Shows an end-to-end workflow that combines product design, model integration, and quantitative evaluation |

## What this project demonstrates

Many LLM demos stop at code generation. This project goes one step further by connecting generation to a usable decision loop. A user can describe a strategy, inspect the translated logic, and then evaluate it quantitatively on real price data. That makes the repository a small but clear example of **LLM-assisted analysis tooling** rather than a toy chat interface.

The project also shows a pragmatic engineering style. Instead of requiring a large framework or broker integration, it keeps the workflow local and inspectable: CSVs go in, strategy code is generated under a constrained format, and the app returns standard backtesting metrics together with an equity curve and detailed trade log.

| Capability area | How it appears here |
|---|---|
| LLM product integration | Natural-language prompts are converted into structured strategy code |
| Quantitative evaluation | Strategies are scored with return, CAGR, drawdown, Sharpe, and win-rate metrics |
| Usable interface design | Dataset selection, date filters, code preview, and result views are all exposed in a simple web UI |
| Rapid prototyping | The repository packages generation, execution, and reporting in a compact single-project application |

## Core features

| Feature | Why it matters |
|---|---|
| Natural-language strategy input | Lets non-programmers express hypotheses without writing indicators by hand |
| Generated Python logic | Makes the translation from prompt to executable rule transparent |
| CSV-based backtesting | Keeps the app flexible and easy to test on custom datasets |
| Date-range filtering | Supports focused evaluation without editing source files |
| Performance metrics | Summarizes outcome quality beyond raw return |
| Equity curve and trade log | Makes behavior inspectable rather than opaque |
| Token usage counter | Surfaces the operational cost of generation during a session |

## Running locally

Install dependencies, add a model key, and start the local server.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
GEMINI_KEY="your-key-here"
GEMINI_MODEL="gemini-2.5-flash"
```

Then launch the app:

```bash
python main.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Data format

The app reads one or more CSV files from the `data/` directory. At minimum, each dataset must contain `Date` and `Close` columns.

| Column | Requirement | Notes |
|---|---|---|
| `Date` | Required | Any format pandas can parse |
| `Close` | Required | Adjusted or standard close price |
| `Open`, `High`, `Low`, `Volume` | Optional | Ignored unless strategy logic explicitly uses them |

Example format:

```csv
Date,Open,High,Low,Close,Volume
2020-01-02,296.24,300.60,293.98,300.35,33870100
2020-01-03,297.15,300.58,296.50,297.43,36580700
```

The filename becomes the dataset label shown in the interface.

## Typical usage flow

A normal session follows a short loop. The user selects one or more datasets, optionally restricts the date range, enters a strategy description, and either previews the generated code or runs the backtest directly. The application then returns a strategy explanation, performance metrics, an equity curve, and the trade history.

| Step | User action |
|---|---|
| 1 | Choose one or more CSV datasets |
| 2 | Optionally set a backtest window |
| 3 | Enter a strategy in natural language |
| 4 | Preview the generated code or run the backtest |
| 5 | Review metrics, charts, and trade-level output |

Example prompts:

```text
Buy when the 20-day moving average crosses above the 50-day moving average, and sell when it crosses back below.
```

```text
Buy when price is 15% above the 90-day low, then exit after a 10% trailing drawdown or after 60 trading days.
```

```text
Buy on new 52-week highs and sell if price falls 8% below the breakout level.
```

## How it works internally

The application asks the model to return structured strategy output, including a `build(df)` function defined over a pandas DataFrame. That code is parsed and executed in a constrained environment, after which the resulting signal stream drives a straightforward cash-and-shares simulator. The output is then rendered back to the interface as metrics, chart data, and trade logs.

This design keeps the generated logic inspectable. The user is not simply asked to trust the model; they can examine the produced strategy code before or alongside the backtest.

| Internal component | Role |
|---|---|
| Prompt-to-code translation | Converts plain-English rules into executable strategy logic |
| Constrained execution path | Loads generated code with limited available objects |
| Backtest simulator | Applies signals to historical data and tracks portfolio state |
| Reporting layer | Returns summary metrics, equity curves, and trade records |

## Project structure

```text
backtester/
├── data/               # Local CSV datasets for testing strategies
├── main.py             # Server, UI, generation flow, and backtest logic
├── requirements.txt
└── .env                # Local API configuration (gitignored)
```

## Reading this repository as a portfolio piece

From a portfolio perspective, **backtester** showcases an engineering pattern that appears in many practical AI products: let a user describe intent in natural language, translate that intent into structured executable logic, and then close the loop with quantitative feedback. The repository is intentionally compact, but it demonstrates product thinking, model integration, and evaluation design in a way that is easy to inspect.
