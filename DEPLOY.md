# Deploying the TCA dashboard

The dataset is **hardcoded** (`data_trades_new2.csv`) and auto-loads — no upload step, ready
for a demo.

## Run locally (what to type)
```powershell
cd "C:\Users\SabinaYessaly\OneDrive - CIGP SA\Desktop\Equity_TCA_Stats"
.venv\Scripts\streamlit run app.py
```
Opens at `http://localhost:8501`. Stop with `Ctrl+C`.

## Share on the local network (same office)
```powershell
.venv\Scripts\streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Others reach it at `http://<your-machine-ip>:8501`.

## Deploy publicly — Streamlit Community Cloud (free)
1. Put this folder in a GitHub repo (must include `app.py`, the `tca/` package,
   `data_trades_new2.csv`, `requirements.txt`, `.streamlit/config.toml`).
2. Go to https://share.streamlit.io → **New app** → pick the repo/branch → main file `app.py`.
3. Deploy. It installs `requirements.txt` and serves a public URL.

Files that MUST ship with the app: `app.py`, `tca/`, `data_trades_new2.csv`,
`180days_child_order_data.csv` (optional drill-down), `requirements.txt`, `.streamlit/config.toml`.

## Notes
- First load runs the full pipeline (~15–25 s, the quantile regression is the slow part); it is
  cached thereafter via `@st.cache_data`.
- To swap the dataset, change `DATA_FILE` at the top of `app.py` (and ship the new CSV).
- `.xlsx` sources need `openpyxl`; the app reads the pre-exported CSV so it is not required.
