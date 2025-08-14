from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import os
import json
from pathlib import Path

# --- Helper functions copied from news.py ---
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# ---------- Config (tweak as needed) ----------
MIN_LIQUID_VOLUME = 100_000
TOP_N_GAINERS = 5
TOP_N_LOSERS = 5
TOP_N_ACTIVE = 5
DATE_FORMAT = "%Y-%m-%d"

# ---------- Gemini (optional but recommended) ----------
def maybe_init_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_APIKEY") or os.getenv("GEMINI")
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        print(f"[WARN] Gemini not initialized: {e}")
        return None

    def generate_with_gemini(prompt: str, system_tone: str = "news"):
        try:
            res = model.generate_content(
                f"""
You are a professional financial markets journalist writing for a Pakistan-focused audience (PSX).
Write in crisp, energetic newsroom style, but stay factual and avoid hype or speculation.
Keep numbers accurate, include PKR prices and percentages where relevant.

PROMPT FACTS:
{prompt}

Rules:
- Keep it concise but vivid.
- Prefer short paragraphs or bullet points.
- Avoid invented reasons ("on rumors") unless provided in facts.
- Use symbols and sector names as provided.
                """.strip()
            )
            text = getattr(res, "text", "").strip()
            return text or ""
        except Exception as e:
            print(f"[WARN] Gemini generation failed: {e}")
            return ""
    return generate_with_gemini

# ---------- Data helpers ----------
def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).replace(",", "").replace("", "").strip()
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return np.nan

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["LDCP", "OPEN", "HIGH", "LOW", "CURRENT", "CHANGE"]:
        df[col] = df[col].apply(to_float)

    df["CHANGE (%)"] = df["CHANGE (%)"].apply(to_float)
    df["VOLUME"] = df["VOLUME"].apply(lambda v: int(float(str(v).replace(",", "").strip())) if str(v).strip() not in ("", "nan", "None") else 0)

    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df["SECTOR"] = df["SECTOR"].astype(str).str.strip()

    df = df.dropna(subset=["SYMBOL", "CURRENT"])
    return df

def rankers(df: pd.DataFrame):
    liquid = df[df["VOLUME"] >= MIN_LIQUID_VOLUME].copy()
    if liquid.empty:
        liquid = df.copy()

    gainers = liquid.sort_values("CHANGE (%)", ascending=False).head(TOP_N_GAINERS)
    losers  = liquid.sort_values("CHANGE (%)", ascending=True).head(TOP_N_LOSERS)
    active  = df.sort_values("VOLUME", ascending=False).head(TOP_N_ACTIVE)
    return gainers, losers, active

def market_breadth(df: pd.DataFrame):
    adv = int((df["CHANGE (%)"] > 0).sum())
    dec = int((df["CHANGE (%)"] < 0).sum())
    unch = int((df["CHANGE (%)"] == 0).sum() + df["CHANGE (%)"].isna().sum())
    avg_chg = float(df["CHANGE (%)"].mean(skipna=True)) if not df["CHANGE (%)"].dropna().empty else 0.0
    total_vol = int(df["VOLUME"].sum())
    return {"advancers": adv, "decliners": dec, "unchanged": unch, "avg_change_pct": round(avg_chg, 2), "total_volume": total_vol}

def sector_stats(df: pd.DataFrame):
    out = []
    for sec, g in df.groupby("SECTOR"):
        avg_pct = round(float(g["CHANGE (%)"].mean(skipna=True)), 2) if not g["CHANGE (%)"].dropna().empty else 0.0
        total_vol = int(g["VOLUME"].sum())
        leader = g.sort_values("CHANGE (%)", ascending=False).head(1)
        lagger = g.sort_values("CHANGE (%)", ascending=True).head(1)
        out.append({
            "sector": sec,
            "average_change_pct": avg_pct,
            "total_volume": total_vol,
            "leader": row_compact(leader.iloc[0]) if not leader.empty else None,
            "laggard": row_compact(lagger.iloc[0]) if not lagger.empty else None,
        })
    out = sorted(out, key=lambda s: s["average_change_pct"], reverse=True)
    return out

def row_compact(row):
    return {
        "symbol": row["SYMBOL"],
        "sector": row["SECTOR"],
        "current": safe_round(row["CURRENT"]),
        "change": safe_round(row["CHANGE"]),
        "change_pct": safe_round(row["CHANGE (%)"]),
        "volume": int(row["VOLUME"]),
    }

def safe_round(v, n=2):
    try:
        return round(float(v), n)
    except Exception:
        return v

def human_price(v):
    try:
        return f"PKR {float(v):,.2f}"
    except Exception:
        return str(v)

def human_pct(v):
    try:
        return f"{float(v):.2f}%"
    except Exception:
        return str(v)

def human_int(v):
    try:
        return f"{int(v):,}"
    except Exception:
        return str(v)

def compose_daily_summary_facts(date_str, breadth, gainers, losers, active, sectors_top3):
    facts = {
        "date": date_str,
        "advancers": breadth["advancers"],
        "decliners": breadth["decliners"],
        "unchanged": breadth["unchanged"],
        "avg_change_pct": breadth["avg_change_pct"],
        "total_volume": breadth["total_volume"],
        "top_gainers": [row_compact(r) for _, r in gainers.iterrows()],
        "top_losers": [row_compact(r) for _, r in losers.iterrows()],
        "most_active": [row_compact(r) for _, r in active.iterrows()],
        "sectors_top": sectors_top3
    }
    return facts

def render_daily_summary_text(facts):
    lines = []
    lines.append(f"PSX Market Wrap — {facts['date']}")
    lines.append(
        f"Market breadth: {facts['advancers']} advancers vs {facts['decliners']} decliners "
        f"({facts['unchanged']} unchanged). Average move: {human_pct(facts['avg_change_pct'])}. "
        f"Total volume: {human_int(facts['total_volume'])} shares."
    )

    if facts["top_gainers"]:
        tg = ", ".join([f"{x['symbol']} ({human_pct(x['change_pct'])} to {human_price(x['current'])})" for x in facts["top_gainers"]])
        lines.append(f"Top gainers: {tg}.")

    if facts["top_losers"]:
        tl = ", ".join([f"{x['symbol']} ({human_pct(x['change_pct'])} to {human_price(x['current'])})" for x in facts["top_losers"]])
        lines.append(f"Top losers: {tl}.")

    if facts["most_active"]:
        ma = ", ".join([f"{x['symbol']} ({human_int(x['volume'])} shares)" for x in facts["most_active"]])
        lines.append(f"Most active by volume: {ma}.")

    if facts["sectors_top"]:
        sec_bits = []
        for s in facts["sectors_top"]:
            sec_bits.append(f"{s['sector']} ({human_pct(s['average_change_pct'])})")
        lines.append("Sector leaders: " + ", ".join(sec_bits) + ".")

    return "\n".join(lines)

def render_stock_blurb(row):
    direction = "rose" if (row["CHANGE (%)"] or 0) > 0 else ("fell" if (row["CHANGE (%)"] or 0) < 0 else "closed flat")
    pct = human_pct(row["CHANGE (%)"]) if pd.notna(row["CHANGE (%)"]) else "0.00%"
    chg = safe_round(row["CHANGE"])
    price = human_price(row["CURRENT"])
    vol = human_int(row["VOLUME"])
    return f"{row['SYMBOL']} {direction} {pct} to {price} with {vol} shares traded."

def render_sector_summary(sec_name, stats):
    avg = human_pct(stats["average_change_pct"])
    leader = stats["leader"]
    lagger = stats["laggard"]
    parts = [f"{sec_name} sector moved {avg} on average"]
    if leader:
        parts.append(f"leader: {leader['symbol']} ({human_pct(leader['change_pct'])})")
    if lagger:
        parts.append(f"laggard: {lagger['symbol']} ({human_pct(lagger['change_pct'])})")
    return ", ".join(parts) + "."
# --- End of helper functions copied from news.py ---

app = FastAPI()

EXCEL_FILE_PATH = "stocks_data.xlsx" # Adjust path as necessary
gemini_client = maybe_init_gemini()

@app.get("/")
async def root():
    return {"message": "Welcome to the PSX Market News API. Use /docs for API documentation."}

@app.get("/news/daily_summary")
async def get_daily_summary_news():
    try:
        df = pd.read_excel(EXCEL_FILE_PATH)
        df = clean_dataframe(df)

        breadth = market_breadth(df)
        gainers, losers, active = rankers(df)
        sectors = sector_stats(df)
        sectors_top3 = sectors[:3] if sectors else []

        date_str = datetime.now().strftime(DATE_FORMAT)
        facts = compose_daily_summary_facts(date_str, breadth, gainers, losers, active, sectors_top3)

        if gemini_client:
            prompt = json.dumps(facts, indent=2)
            enhanced_daily = gemini_client(prompt)
            if not enhanced_daily.strip():
                enhanced_daily = render_daily_summary_text(facts)
        else:
            enhanced_daily = render_daily_summary_text(facts)

        return {
            "date": date_str,
            "market_overview": breadth,
            "daily_summary_text": enhanced_daily,
            "top_gainers": facts["top_gainers"],
            "top_losers": facts["top_losers"],
            "most_active": facts["most_active"],
            "sectors_top3": sectors_top3
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/symbol/{symbol}")
async def get_symbol_news(symbol: str):
    try:
        df = pd.read_excel(EXCEL_FILE_PATH)
        df = clean_dataframe(df)
        
        symbol_upper = symbol.upper()
        stock_data = df[df["SYMBOL"] == symbol_upper]

        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")

        row = stock_data.iloc[0]
        blurb = render_stock_blurb(row)

        return {
            "symbol": row["SYMBOL"],
            "sector": row["SECTOR"],
            "current_price": safe_round(row["CURRENT"]),
            "change": safe_round(row["CHANGE"]),
            "change_pct": safe_round(row["CHANGE (%)"]),
            "volume": int(row["VOLUME"]),
            "news_blurb": blurb
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/sector/{sector_name}")
async def get_sector_news(sector_name: str):
    try:
        df = pd.read_excel(EXCEL_FILE_PATH)
        df = clean_dataframe(df)

        sector_data = df[df["SECTOR"].str.contains(sector_name, case=False, na=False)]

        if sector_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for sector: {sector_name}")
        
        # Calculate stats for the requested sector
        # sector_group = df.groupby("SECTOR").get_group(sector_name) # This line might cause an error if sector_name is not exact
        # Instead, filter based on the sector_data found
        # For accurate sector stats, we need to make sure the sector_name matches exactly or is handled consistently
        
        # Let's refine the sector_stats call to use the filtered sector_data and then extract the matching sector
        all_sector_stats = sector_stats(df) # Calculate for all sectors

        # Find the specific sector in the list of calculated stats
        found_sector_stats = next((s for s in all_sector_stats if s["sector"].lower() == sector_name.lower()), None)

        if not found_sector_stats:
             raise HTTPException(status_code=404, detail=f"Detailed stats not found for sector: {sector_name}")

        return {
            "sector": found_sector_stats["sector"],
            "average_change_pct": found_sector_stats["average_change_pct"],
            "total_volume": found_sector_stats["total_volume"],
            "leader": found_sector_stats["leader"],
            "laggard": found_sector_stats["laggard"],
            "summary": render_sector_summary(found_sector_stats["sector"], found_sector_stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
