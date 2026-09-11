# Getting a real LOBSTER sample

LOBSTER publishes free single-day samples at
<https://lobsterdata.com/info/DataSamples.php>. They are not redistributed here,
which is why this repository ships a synthetic generator instead.

Download a message/orderbook pair — for example `AAPL_2012-06-21`, level 10 —
unzip it into a `data/` directory (already gitignored), and run:

```bash
PYTHONPATH=src python3 -m obrp.run \
  --message-file   data/AAPL_2012-06-21_34200000_57600000_message_10.csv \
  --orderbook-file data/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv \
  --levels 10
```

The two files must be the matched pair for the same ticker, date and depth —
they are aligned row by row, and the loader raises if the lengths differ.
