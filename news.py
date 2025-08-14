#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated PSX Market News Generator
-----------------------------------
Reads an Excel file with columns:
SYMBOL, SECTOR, LISTED IN, LDCP, OPEN, HIGH, LOW, CURRENT, CHANGE, CHANGE (%), VOLUME

Outputs:
- market_news.json  (consolidated feed for your website)
- Optional: per-symbol and per-sector JSON files

Gemini is used to transform structured facts into exciting, newsroom-style copy.
If Gemini is unavailable, the script falls back to high-quality templated text.

Usage:
    python generate_market_news.py --excel stocks_data.xlsx --out out_dir

Schedule (Linux cron example, runs at 5:05 PM daily):
    5 17 * * 1-5 /usr/bin/python3 /path/generate_market_news.py --excel /path/stocks_data.xlsx --out /path/www/news

Environment:
    .env must contain: GEMINI_API_KEY=your_key_here
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---------- Config (tweak as needed) ----------
MIN_LIQUID_VOLUME = 100_000          # ignore micro-cap noise for top gainers/losers rankings
TOP_N_GAINERS = 5
TOP_N_LOSERS = 5
TOP_N_ACTIVE = 5
WRITE_PER_SYMBOL = True              # set False if you don't want per-symbol JSON files
WRITE_PER_SECTOR = True              # set False if you don't want per-sector JSON files
OUTPUT_MAIN_FILENAME = "market_news.json"
DATE_FORMAT = "%Y-%m-%d"

# ---------- Gemini (optional but recommended) ----------
def maybe_init_gemini():
    """
    Initializes Gemini client if GEMINI_API_KEY exists.
    Returns a function: generate_with_gemini(prompt) -> string
    or None if not available.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_APIKEY") or os.getenv("GEMINI")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        print(f"[WARN] Gemini not initialized: {e}")
        return None

    def generate_with_gemini(prompt: str, system_tone: str = "news"):
        try:
            # Keep prompts structured, ask for concise, punchy newsroom tone.
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
            # Gemini sometimes returns empty text if safety blocked; fallback handled by caller.
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
    s = str(x).replace(",", "").replace("�", "").strip()
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return np.nan

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns
    expected_cols = ["SYMBOL", "SECTOR", "LISTED IN", "LDCP", "OPEN", "HIGH", "LOW", "CURRENT", "CHANGE", "CHANGE (%)", "VOLUME"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Excel missing columns: {missing}")

    df = df.copy()
    # Numeric conversions
    for col in ["LDCP", "OPEN", "HIGH", "LOW", "CURRENT", "CHANGE"]:
        df[col] = df[col].apply(to_float)

    df["CHANGE (%)"] = df["CHANGE (%)"].apply(to_float)
    df["VOLUME"] = df["VOLUME"].apply(lambda v: int(float(str(v).replace(",", "").strip())) if str(v).strip() not in ("", "nan", "None") else 0)

    # SYMBOL & SECTOR cleanup
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df["SECTOR"] = df["SECTOR"].astype(str).str.strip()

    # Drop rows missing critical fields
    df = df.dropna(subset=["SYMBOL", "CURRENT"])
    return df

def rankers(df: pd.DataFrame):
    # Liquidity filter for ranks
    liquid = df[df["VOLUME"] >= MIN_LIQUID_VOLUME].copy()
    if liquid.empty:  # fallback if everything is illiquid
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
    # sort by strongest sectors
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

# ---------- Templated text (fallback or base for Gemini) ----------
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
    # Clean, energetic newsroom tone (fallback if Gemini not used)
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

# ---------- Main generation ----------
def main():
    parser = argparse.ArgumentParser(description="Generate automated PSX news from Excel.")
    parser.add_argument("--excel", required=True, help="Path to Excel file (stocks_data.xlsx)")
    parser.add_argument("--out", required=True, help="Output directory for JSON")
    parser.add_argument("--date", default=datetime.now().strftime(DATE_FORMAT), help="Override date (YYYY-MM-DD)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & clean
    df = pd.read_excel(args.excel)
    df = clean_dataframe(df)

    # Core stats
    breadth = market_breadth(df)
    gainers, losers, active = rankers(df)
    sectors = sector_stats(df)
    sectors_top3 = sectors[:3] if sectors else []

    # Compose facts
    facts = compose_daily_summary_facts(args.date, breadth, gainers, losers, active, sectors_top3)

    # Optional Gemini enhancement
    gemini = maybe_init_gemini()
    if gemini:
        # Build a concise, structured prompt from facts
        prompt = json.dumps(facts, indent=2)
        enhanced_daily = gemini(prompt)
        if not enhanced_daily.strip():
            enhanced_daily = render_daily_summary_text(facts)
    else:
        enhanced_daily = render_daily_summary_text(facts)

    # Per-stock blurbs (and optional Gemini micro-polish in batches)
    stock_items = []
    for _, r in df.iterrows():
        base_blurb = render_stock_blurb(r)
        stock_items.append({
            "symbol": r["SYMBOL"],
            "sector": r["SECTOR"],
            "current": safe_round(r["CURRENT"]),
            "change": safe_round(r["CHANGE"]),
            "change_pct": safe_round(r["CHANGE (%)"]),
            "volume": int(r["VOLUME"]),
            "blurb": base_blurb
        })

    # Per-sector summaries
    sector_items = []
    for s in sectors:
        sector_items.append({
            "sector": s["sector"],
            "average_change_pct": s["average_change_pct"],
            "total_volume": s["total_volume"],
            "leader": s["leader"],
            "laggard": s["laggard"],
            "summary": render_sector_summary(s["sector"], s)
        })

    # Build consolidated JSON
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": args.date,
        "market_overview": breadth,
        "daily_summary_text": enhanced_daily,
        "top_gainers": [row_compact(r) for _, r in gainers.iterrows()],
        "top_losers": [row_compact(r) for _, r in losers.iterrows()],
        "most_active": [row_compact(r) for _, r in active.iterrows()],
        "sectors": sector_items,
        "stocks": stock_items
    }

    # Write main feed
    main_path = out_dir / OUTPUT_MAIN_FILENAME
    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {main_path}")

    # Optional: per-symbol JSON (for your “news by stock” pages)
    if WRITE_PER_SYMBOL:
        sym_dir = out_dir / "stocks"
        sym_dir.mkdir(exist_ok=True)
        for item in stock_items:
            p = sym_dir / f"{item['symbol']}.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)

    # Optional: per-sector JSON (for your “news by sector” pages)
    if WRITE_PER_SECTOR:
        sec_dir = out_dir / "sectors"
        sec_dir.mkdir(exist_ok=True)
        for item in sector_items:
            # safe filename
            name = "".join(ch for ch in item["sector"] if ch.isalnum() or ch in ("-", "_")).strip() or "Unknown"
            p = sec_dir / f"{name}.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
