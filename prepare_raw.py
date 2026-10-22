"""Срез Yambda-50M под CPU: дни 180-300 и топ-20k треков по train-периоду (дни 180-280).

python prepare_raw.py /path/to/yambda/flat/50m   ->  data/{listens,likes,dislikes}.parquet
"""
import sys
from pathlib import Path

import polars as pl

DAY = 86400
T0, TTRAIN, VOCAB = 180 * DAY, 280 * DAY, 20000
src, dst = Path(sys.argv[1]), Path(__file__).resolve().parent / "data"
dst.mkdir(exist_ok=True)
s = pl.scan_parquet(src / "listens.parquet").filter(pl.col("timestamp") >= T0)
top = s.filter(pl.col("timestamp") < TTRAIN).group_by("item_id").len().sort("len", descending=True).head(VOCAB).collect()
s.join(top.lazy().select("item_id"), on="item_id", how="semi").sort(["uid", "timestamp"]).collect() \
    .write_parquet(dst / "listens.parquet", compression="zstd")
for f in ["likes", "dislikes"]:
    pl.read_parquet(src / f"{f}.parquet").filter(pl.col("timestamp") >= T0) \
        .join(top.select("item_id"), on="item_id", how="semi").write_parquet(dst / f"{f}.parquet", compression="zstd")
print("ok")
