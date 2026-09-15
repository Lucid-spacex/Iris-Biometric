# Iris Biometric Identification — Live Demo

This folder is a ready-to-deploy web app for the iris biometric
identification pipeline (segmentation → Daugman normalisation → Log-Gabor
encoding → AES-256 encrypted template storage → Hamming/Euclidean one-to-many
matching). It runs the real code, not a mockup — because CASIA-Iris-Thousand
access is still pending, the enrolled "gallery" it demos against is built
from generated synthetic eye images, clearly labelled as such in the UI.

## What your supervisor will see

A single page where they can click **"Scan a genuine probe"** (a re-scan of
someone already enrolled — should match), **"Scan an impostor probe"** (a new
identity never enrolled — should be rejected), or upload their own photo. The
page shows the actual detected iris boundaries, the unwrapped strip, the
resulting binary IrisCode, the match/no-match verdict from both distance
metrics, and a snippet of the AES-encrypted template as stored.

---

## Deploying it — no coding required, about 10 minutes

We'll use **Render** (free tier, no credit card required for this). Render
builds and hosts the app for you and gives you a public `https://` link.

### Step 1 — Put the code on GitHub

1. Go to [github.com](https://github.com) and create a free account if you
   don't have one.
2. Click the **+** in the top right → **New repository**. Name it
   `iris-biometric-demo`, keep it **Public**, click **Create repository**.
3. On the next page, click **uploading an existing file**.
4. Drag in *every file and folder* from this `deploy` folder (keep the
   `iris_biometric`, `templates`, and `static` folders intact — don't rename
   anything).
5. Scroll down, click **Commit changes**.

### Step 2 — Deploy on Render

1. Go to [render.com](https://render.com) and sign up (you can sign up
   directly with your GitHub account — this also connects the two for you).
2. Click **New +** → **Web Service**.
3. Choose **Build and deploy from a Git repository**, then select the
   `iris-biometric-demo` repository you just created.
4. Render will detect it's a Python app. Leave the settings as suggested,
   but double check:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. **Add Environment Variables (Neon Postgres):**
   - Under the **Environment Variables** section on Render, add:
     - `DATABASE_URL`: paste your Neon Postgres connection string
       (`postgresql://...`)
     - `AES_MASTER_KEY_HEX`: (optional) paste your 64-character AES hex key so
       templates can be decrypted across restarts.
6. Click **Deploy Web Service**.
7. Wait 2–5 minutes while it builds. When it says **Live**, Render shows you a
   link like `https://iris-biometric-demo.onrender.com`.

---

## Task 1: Setting up Neon Postgres Database (Free)

1. Sign up for free at [neon.tech](https://neon.tech).
2. Click **Create Project** (name it `iris-biometric-db`).
3. On your Neon dashboard, copy your **Connection String**. It looks like:
   `postgresql://neondb_owner:password@ep-xyz.us-east-2.aws.neon.tech/neondb?sslmode=require`
4. In your project folder, open `.env` and paste:
   ```env
   DATABASE_URL=your_neon_connection_string_here
   ```
5. Tables (`encrypted_templates` and `gallery_projector`) are created
   automatically on first launch!
   - Privacy guarantee: No plaintext names or identity strings are ever stored.
     Only random 128-bit reference IDs, AES-256 ciphertext, and IV bytes.

---

## Task 2: Running MMU1 Benchmark Evaluation

1. Download the MMU1 dataset (~14 MB) from Kaggle:
   `https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset`
2. Run the helper to auto-detect and extract the zip:
   ```bash
   python download_mmu.py
   ```
3. Run the full biometric evaluation pipeline:
   ```bash
   python evaluate_mmu.py
   ```
   This will:
   - Partition the dataset (enrolling 1 image per eye into the gallery).
   - Evaluate all remaining images as genuine and impostor probes.
   - Calculate real Equal Error Rate (EER) operating thresholds.
   - Output Accuracy, Precision, Recall, F1, FAR, and FRR.
   - Generate `mmu1_evaluation_report.json` for your seminar report.

---

## Running locally on your own laptop

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` in your browser.
