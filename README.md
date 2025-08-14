# PSX Market News API

This project provides a FastAPI application to generate automated news and insights from Pakistan Stock Exchange (PSX) data, based on an Excel file input. It offers APIs for daily market summaries, symbol-specific news, and sector-wise performance.

## Setup and Running the API

Follow these steps to set up and run the API server locally:

1.  **Navigate to the `scrap` directory:**
    ```bash
    cd scrap
    ```

2.  **Install the dependencies:**
    Ensure you have Python 3 and pip installed. Then, install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    If `pip` is not recognized, try:
    ```bash
    python3 -m pip install -r requirements.txt
    ```

3.  **Place your `stocks_data.xlsx` file:**
    Make sure your Excel data file named `stocks_data.xlsx` is present in the `scrap/` directory, alongside `main.py` and `requirements.txt`.

4.  **Run the FastAPI application:**
    ```bash
    uvicorn main:app --reload
    ```
    The API server will start, usually on `http://127.0.0.1:8000`.

## API Endpoints

Once the server is running, you can access the API documentation at `http://127.0.0.1:8000/docs` for an interactive interface to test the endpoints. Below are the available endpoints:

### 1. Root Endpoint
- **Method:** `GET`
- **URL:** `/`
- **Description:** Returns a welcome message and directs to API documentation.

### 2. Daily Market Summary News
- **Method:** `GET`
- **URL:** `/news/daily_summary`
http://127.0.0.1:8000/news/daily_summary
- **Description:** Retrieves a comprehensive daily summary of the stock market, including market breadth (advancers, decliners, unchanged), average change percentage, total volume, top gainers, top losers, and most active stocks.

### 3. Symbol-based Stock News
- **Method:** `GET`
- **URL:** `/news/symbol/{symbol}`
- **Description:** Fetches detailed news and data for a specific stock using its trading symbol.
- **Example:** `http://127.0.0.1:8000/news/symbol/LUCK` (Replace `LUCK` with an actual stock symbol from your `stocks_data.xlsx`)

### 4. Sector-wise News
- **Method:** `GET`
- **URL:** `/news/sector/{sector_name}`
- **Description:** Provides an overview and statistics for a specific market sector, including average change, total volume, and identified leader/laggard stocks within that sector.
- **Example:** `http://127.0.0.1:8000/news/sector/Cement` (Replace `Cement` with an actual sector name from your `stocks_data.xlsx`)

## Gemini API Key (Optional)

If you wish to use the Gemini integration for enhanced news generation, create a `.env` file in the `scrap/` directory with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

If the `GEMINI_API_KEY` is not provided, the script will fall back to high-quality templated text for news generation.

### Scraping file
s.py file should be run to fetch data and store in stocks_data.xlsx file for latest updated data.