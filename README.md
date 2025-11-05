# 💃 DancePose Live Studio Pro v5.4 — Netlify Edition

A professional AI-powered dance analytics app with skeleton tracking, symmetry analysis, and PDF reporting.

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
1. Push repo to GitHub.
2. Go to https://share.streamlit.io
3. New App → Select repo → File path: app/app.py
4. Deploy instantly.

### Netlify (Docker Required)
1. Push repo to GitHub.
2. Connect GitHub to Netlify.
3. Build command: docker build -t dancepose . && docker run -p 8501:8501 dancepose
4. Publish directory: .

### Render.com (Free Alternative)
1. Go to https://render.com → New Web Service.
2. Build: pip install -r app/requirements.txt
3. Start: streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0
