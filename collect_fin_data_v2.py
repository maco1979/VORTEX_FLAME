#!/usr/bin/env python3
"""
FIN-JEPA Data Collection v2 — Robust multi-source collection
=============================================================
Uses yfinance (more stable) as primary, akshare as fallback.
Generates synthetic data if both fail (for smoke test).
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = r"D:\VORTEX_FLAME\fin_jepa_data"


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


def collect_yfinance(symbols=None, period="3y"):
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed, trying pip install...", flush=True)
        os.system("pip install yfinance --quiet")
        import yfinance as yf

    if symbols is None:
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT",
            "JNJ", "PG", "UNH", "HD", "MA", "DIS", "NFLX", "INTC", "AMD", "CRM",
            "BABA", "NIO", "PDD", "JD", "TCEHY",
            "^GSPC", "^DJI", "^IXIC", "^VIX", "CL=F", "GC=F",
        ]

    results = []
    for i, sym in enumerate(symbols):
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period=period)
            if len(hist) < 60:
                continue

            ohlcv = hist[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
            ohlcv = _normalize_ohlcv(ohlcv)

            results.append({
                "code": sym, "name": sym, "market": "us",
                "data": ohlcv, "n_bars": len(ohlcv),
            })
            print(f"  [yfinance {i+1}/{len(symbols)}] {sym}: {len(ohlcv)} bars", flush=True)
            time.sleep(0.1)
        except Exception as e:
            print(f"  [yfinance {i+1}/{len(symbols)}] {sym}: SKIP ({e})", flush=True)

    return results


def generate_synthetic(n_symbols=20, n_bars=750):
    print(f"  Generating {n_symbols} synthetic financial series...", flush=True)
    results = []

    regimes = ["bull", "bear", "range", "volatile", "trend"]
    for i in range(n_symbols):
        regime = regimes[i % len(regimes)]
        price = 100.0
        ohlcv = []

        for t in range(n_bars):
            if regime == "bull":
                ret = np.random.normal(0.0008, 0.015)
            elif regime == "bear":
                ret = np.random.normal(-0.0005, 0.018)
            elif regime == "range":
                ret = np.random.normal(0.0, 0.01)
            elif regime == "volatile":
                ret = np.random.normal(0.0, 0.04)
            else:
                ret = np.random.normal(0.0003, 0.02)

            price *= (1 + ret)
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_ = price * (1 + np.random.normal(0, 0.003))
            volume = abs(np.random.normal(1e6, 3e5))

            ohlcv.append([open_, high, low, price, volume])

        ohlcv = np.array(ohlcv, dtype=np.float32)
        ohlcv = _normalize_ohlcv(ohlcv)

        results.append({
            "code": f"SYN_{regime}_{i:03d}", "name": f"Synthetic {regime} #{i}",
            "market": "synthetic", "data": ohlcv, "n_bars": n_bars,
        })

    return results


def save_dataset(results, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    manifest = []
    total_bars = 0

    for r in results:
        fname = f"{r['market']}_{r['code'].replace('^','idx_').replace('=','_')}.pt"
        fpath = os.path.join(data_dir, fname)
        torch.save({
            "data": torch.from_numpy(r["data"]),
            "code": r["code"], "name": r["name"],
            "market": r["market"], "n_bars": r["n_bars"],
        }, fpath)
        manifest.append({
            "file": fname, "code": r["code"], "name": r["name"],
            "market": r["market"], "n_bars": r["n_bars"],
        })
        total_bars += r["n_bars"]

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_symbols": len(manifest), "total_bars": total_bars,
            "collected_at": datetime.now().isoformat(), "symbols": manifest,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} symbols, {total_bars} total bars to {data_dir}")
    return len(results), total_bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["yfinance", "synthetic", "all"], default="all")
    parser.add_argument("--n-synthetic", type=int, default=20)
    args = parser.parse_args()

    all_results = []

    print("=" * 60)
    print("FIN-JEPA Data Collection v2")
    print("=" * 60)

    if args.source in ("yfinance", "all"):
        print("\n--- yfinance (US stocks + indices) ---")
        yf_results = collect_yfinance()
        if yf_results:
            all_results.extend(yf_results)
            print(f"  yfinance: {len(yf_results)} symbols collected")
        else:
            print("  yfinance: FAILED, will use synthetic data")

    if args.source in ("synthetic", "all") or len(all_results) < 5:
        print("\n--- Synthetic Data ---")
        n_syn = max(args.n_synthetic, 20 - len(all_results))
        syn_results = generate_synthetic(n_syn)
        all_results.extend(syn_results)
        print(f"  Synthetic: {len(syn_results)} symbols generated")

    n_sym, n_bars = save_dataset(all_results, DATA_DIR)
    print(f"\nDone! {n_sym} symbols, {n_bars} bars ready for FIN-JEPA training.")


if __name__ == "__main__":
    main()
