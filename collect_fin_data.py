#!/usr/bin/env python3
"""
FIN-JEPA (Financial JEPA) Data Collection for VORTEX_FLAME
===========================================================

Collects financial time-series data via AKShare for C-FINJEPA training.

Data sources:
  - A-share daily K-line (沪深A股日线)
  - US stock daily K-line (美股日线)
  - Index daily data (指数日线)
  - Futures daily data (期货日线)
  - Crypto daily data (加密货币日线)

Output format: .pt files with OHLCV tensors
  shape: [T, 5] where 5 = [open, high, low, close, volume]
  normalized per-symbol with z-score

Usage:
  python collect_fin_data.py --symbols 50 --period 3y
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = r"D:\VORTEX_FLAME\fin_jepa_data"


def collect_ashare_daily(n_symbols=50, period_years=3):
    import akshare as ak

    print(f"[A-Share] Fetching stock list...", flush=True)
    df = ak.stock_zh_a_spot_em()
    codes = df["代码"].tolist()[:n_symbols]
    names = df["名称"].tolist()[:n_symbols]

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * period_years)).strftime("%Y%m%d")

    results = []
    for i, (code, name) in enumerate(zip(codes, names)):
        try:
            kline = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq"
            )
            if len(kline) < 60:
                continue

            ohlcv = kline[["开盘", "最高", "最低", "收盘", "成交量"]].values.astype(np.float32)
            ohlcv = _normalize_ohlcv(ohlcv)

            results.append({
                "code": code, "name": name, "market": "ashare",
                "data": ohlcv, "n_bars": len(ohlcv),
            })
            print(f"  [{i+1}/{n_symbols}] {code} {name}: {len(ohlcv)} bars", flush=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [{i+1}/{n_symbols}] {code} {name}: SKIP ({e})", flush=True)
            time.sleep(0.5)

    return results


def collect_us_daily(symbols=None, period_years=3):
    import akshare as ak

    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT",
                    "JNJ", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "NFLX", "INTC"]

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * period_years)).strftime("%Y%m%d")

    results = []
    for i, sym in enumerate(symbols):
        try:
            kline = ak.stock_us_hist(
                symbol=sym, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq"
            )
            if len(kline) < 60:
                continue

            ohlcv = kline[["开盘", "最高", "最低", "收盘", "成交量"]].values.astype(np.float32)
            ohlcv = _normalize_ohlcv(ohlcv)

            results.append({
                "code": sym, "name": sym, "market": "us",
                "data": ohlcv, "n_bars": len(ohlcv),
            })
            print(f"  [US {i+1}/{len(symbols)}] {sym}: {len(ohlcv)} bars", flush=True)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [US {i+1}/{len(symbols)}] {sym}: SKIP ({e})", flush=True)
            time.sleep(1.0)

    return results


def collect_index_daily(period_years=3):
    import akshare as ak

    indices = {
        "000001": "上证指数", "399001": "深证成指", "399006": "创业板指",
        "000300": "沪深300", "000905": "中证500", "000852": "中证1000",
    }

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * period_years)).strftime("%Y%m%d")

    results = []
    for code, name in indices.items():
        try:
            kline = ak.stock_zh_index_daily_em(symbol=code)
            kline = kline[kline["日期"] >= start_date]
            kline = kline[kline["日期"] <= end_date]

            if len(kline) < 60:
                continue

            cols = [c for c in ["开盘", "最高", "最低", "收盘", "成交量"] if c in kline.columns]
            if len(cols) < 4:
                continue
            ohlcv = kline[cols].values.astype(np.float32)
            if ohlcv.shape[1] == 4:
                ohlcv = np.column_stack([ohlcv, np.zeros(len(ohlcv), dtype=np.float32)])
            ohlcv = _normalize_ohlcv(ohlcv)

            results.append({
                "code": code, "name": name, "market": "index",
                "data": ohlcv, "n_bars": len(ohlcv),
            })
            print(f"  [Index] {code} {name}: {len(ohlcv)} bars", flush=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [Index] {code} {name}: SKIP ({e})", flush=True)
            time.sleep(0.5)

    return results


def collect_crypto_daily(period_years=1):
    import akshare as ak

    coins = ["BTC", "ETH", "BNB", "SOL", "ADA", "DOGE", "XRP", "DOT"]

    results = []
    for coin in coins:
        try:
            kline = ak.crypto_hist(symbol=f"{coin}/USDT", period="1d")
            if kline is None or len(kline) < 60:
                continue

            col_map = {}
            for c in kline.columns:
                cl = c.lower()
                if "open" in cl: col_map["open"] = c
                elif "high" in cl: col_map["high"] = c
                elif "low" in cl: col_map["low"] = c
                elif "close" in cl: col_map["close"] = c
                elif "volume" in cl or "vol" in cl: col_map["volume"] = c

            needed = ["open", "high", "low", "close", "volume"]
            if not all(k in col_map for k in needed[:4]):
                continue

            cols = [col_map.get(k) for k in needed if k in col_map]
            ohlcv = kline[cols].values.astype(np.float32)
            if ohlcv.shape[1] == 4:
                ohlcv = np.column_stack([ohlcv, np.zeros(len(ohlcv), dtype=np.float32)])
            ohlcv = _normalize_ohlcv(ohlcv)

            cutoff = datetime.now() - timedelta(days=365 * period_years)
            results.append({
                "code": coin, "name": coin, "market": "crypto",
                "data": ohlcv, "n_bars": len(ohlcv),
            })
            print(f"  [Crypto] {coin}: {len(ohlcv)} bars", flush=True)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [Crypto] {coin}: SKIP ({e})", flush=True)
            time.sleep(1.0)

    return results


def _normalize_ohlcv(ohlcv):
    n = ohlcv.shape[0]
    for col in range(ohlcv.shape[1]):
        vals = ohlcv[:, col]
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            ohlcv[:, col] = 0
            continue
        mean = valid.mean()
        std = valid.std()
        if std < 1e-8:
            ohlcv[:, col] = 0
        else:
            ohlcv[:, col] = (vals - mean) / std
    ohlcv = np.nan_to_num(ohlcv, nan=0.0, posinf=3.0, neginf=-3.0)
    return ohlcv


def save_dataset(results, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    manifest = []
    total_bars = 0

    for r in results:
        fname = f"{r['market']}_{r['code']}.pt"
        fpath = os.path.join(data_dir, fname)
        torch.save({
            "data": torch.from_numpy(r["data"]),
            "code": r["code"],
            "name": r["name"],
            "market": r["market"],
            "n_bars": r["n_bars"],
        }, fpath)
        manifest.append({
            "file": fname, "code": r["code"], "name": r["name"],
            "market": r["market"], "n_bars": r["n_bars"],
        })
        total_bars += r["n_bars"]

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        import json
        json.dump({"total_symbols": len(manifest), "total_bars": total_bars,
                    "collected_at": datetime.now().isoformat(), "symbols": manifest},
                   f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} symbols, {total_bars} total bars to {data_dir}")
    return len(results), total_bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=50, help="Number of A-share symbols")
    parser.add_argument("--period", type=int, default=3, help="Years of history")
    parser.add_argument("--output", type=str, default=DATA_DIR)
    args = parser.parse_args()

    all_results = []

    print("=" * 60)
    print("FIN-JEPA Data Collection")
    print("=" * 60)

    print("\n--- A-Share Daily ---")
    all_results.extend(collect_ashare_daily(args.symbols, args.period))

    print("\n--- US Stock Daily ---")
    all_results.extend(collect_us_daily(period_years=args.period))

    print("\n--- Index Daily ---")
    all_results.extend(collect_index_daily(args.period))

    print("\n--- Crypto Daily ---")
    all_results.extend(collect_crypto_daily(min(args.period, 1)))

    n_sym, n_bars = save_dataset(all_results, args.output)
    print(f"\nDone! {n_sym} symbols, {n_bars} bars collected.")


if __name__ == "__main__":
    main()
