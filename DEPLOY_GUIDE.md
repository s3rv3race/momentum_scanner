# Momentum Scanner Dashboard — PythonAnywhere Deployment Guide

## Quick Setup (10 minutes)

### 1. Upload Files to PythonAnywhere

Go to https://www.pythonanywhere.com → **Files** tab

Upload these 3 files to `/home/YOUR_USERNAME/momentum_scanner/`:
- `flask_app.py`
- `nasdaq_screener.csv`

Then create a subfolder `static/` and upload:
- `static/index.html`

Your structure should look like:
```
/home/YOUR_USERNAME/momentum_scanner/
├── flask_app.py
├── nasdaq_screener.csv
└── static/
    └── index.html
```

### 2. Install Dependencies

Go to **Consoles** → Start a **Bash console**, then run:

```bash
pip3 install --user yfinance flask pandas numpy
```

### 3. Create the Web App

Go to **Web** tab → **Add a new web app**
- Choose **Manual configuration**
- Select **Python 3.10** (or latest available)

### 4. Configure WSGI

On the **Web** tab, click the **WSGI configuration file** link.

**Delete everything** in that file and replace with:

```python
import sys
import os

project_home = '/home/YOUR_USERNAME/momentum_scanner'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.chdir(project_home)

from flask_app import app as application
```

Replace `YOUR_USERNAME` with your actual PythonAnywhere username.

### 5. Set Static Files

On the **Web** tab, under **Static files**, add:
- **URL**: `/static/`
- **Directory**: `/home/YOUR_USERNAME/momentum_scanner/static`

### 6. Reload

Click the green **Reload** button on the Web tab.

### 7. Access from Phone

Your dashboard is live at:
```
https://YOUR_USERNAME.pythonanywhere.com
```

Open this URL on your phone browser. Bookmark it or add to home screen for app-like access.

---

## Add to Home Screen (Phone)

### iPhone (Safari):
1. Open your dashboard URL
2. Tap the Share button (square with arrow)
3. Tap "Add to Home Screen"
4. Name it "Momentum Scan"

### Android (Chrome):
1. Open your dashboard URL
2. Tap the 3-dot menu
3. Tap "Add to Home screen"

---

## Updating the CSV

When you download a fresh `nasdaq_screener.csv` from NASDAQ:
1. Go to PythonAnywhere **Files** tab
2. Navigate to `/home/YOUR_USERNAME/momentum_scanner/`
3. Upload the new `nasdaq_screener.csv` (overwrites the old one)
4. No reload needed — it reads the CSV fresh each scan

---

## Notes

- **Free tier**: PythonAnywhere free accounts include web hosting but may have
  slower performance. Scanning ~200 tickers takes 3-5 minutes.
- **Timeout**: If the scan times out on the free tier, reduce `head(200)` to
  `head(100)` in `flask_app.py` → `load_tickers_from_csv()` function.
- **yfinance rate limits**: If you scan too frequently, Yahoo may throttle.
  Wait at least 5 minutes between scans.
