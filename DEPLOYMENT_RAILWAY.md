# Railway Deployment Guide

This guide explains how to deploy the Image Analyzer to Railway.

## Prerequisites

1. Railway account (sign up at https://railway.app)
2. GitHub repository connected to Railway
3. Environment variables configured

## Quick Start

### 1. Connect Repository to Railway

1. Go to Railway dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your `image-analyzer` repository
5. Railway will auto-detect the `Procfile` in `deployments/railway/`

### 2. Configure Environment Variables

In Railway dashboard, go to your project → Variables tab and add:

```bash
REPLICATE_API_TOKEN=your-replicate-token-here
SECRET_KEY=your-secret-key-here  # Generate with: python3 -c "import secrets; print(secrets.token_hex(32))"
ADMIN_PASSWORD_HASH=your-sha256-hash-here  # Generate with: python3 -c "import hashlib; print(hashlib.sha256(b'your-password').hexdigest())"
PORT=5000
FLASK_ENV=production
```

**Important:** 
- Railway automatically sets `RAILWAY_ENVIRONMENT`, `PORT`, and `RAILWAY_PUBLIC_DOMAIN`
- The app will auto-detect Railway and automatically set `APP_BASE_URL` to `https://RAILWAY_PUBLIC_DOMAIN`
- **You don't need to set `APP_BASE_URL` manually on Railway** - it's auto-configured!

### 3. Configure Build Settings

Railway should auto-detect:
- **Build Command:** (auto-detected from Nixpacks)
- **Start Command:** Uses `Procfile` from `deployments/railway/Procfile`

If needed, manually set:
- **Root Directory:** `/` (root of repo)
- **Build Command:** `pip install -r requirementsDeployment.txt`
- **Start Command:** `gunicorn --worker-class eventlet --workers 1 --bind 0.0.0.0:$PORT --timeout 120 app:app`

### 4. Deploy

1. Railway will automatically deploy on every push to main branch
2. Or click "Deploy" in Railway dashboard
3. Wait for build to complete
4. Your app will be available at: `https://image-analyzer.up.railway.app`

## Railway-Specific Features

### Automatic HTTPS
Railway provides HTTPS automatically via their domain. No SSL certificate setup needed.

### Port Configuration
Railway automatically sets the `PORT` environment variable. The app listens on `0.0.0.0:$PORT`.

### Public Domain
Railway provides a public domain automatically. The app auto-detects this via `RAILWAY_PUBLIC_DOMAIN`.

### Custom Domain
To use a custom domain:
1. Go to Settings → Domains
2. Add your custom domain
3. Set `APP_BASE_URL` environment variable to your custom domain (e.g., `https://yourdomain.com`)
   - This overrides the auto-detected Railway domain

## Troubleshooting

### Build Fails
- Check that `requirementsDeployment.txt` is in the root directory
- Verify Python version (Railway uses Python 3.12 by default)
- Check build logs in Railway dashboard

### App Crashes on Startup
- Verify all environment variables are set
- Check logs: Railway dashboard → Deployments → View Logs
- Ensure `SECRET_KEY` and `ADMIN_PASSWORD_HASH` are set

### Replicate API Errors
- Verify `REPLICATE_API_TOKEN` is set correctly
- Check that `APP_BASE_URL` is set correctly (auto-set on Railway, check logs if issues)
- Replicate requires publicly accessible URLs (Railway provides HTTPS automatically)

### SocketIO Not Working
- Ensure `eventlet` is installed (already in requirementsDeployment.txt)
- Check that Gunicorn uses `eventlet` worker class (configured in Procfile)

## Monitoring

- **Logs:** Railway dashboard → Deployments → View Logs
- **Metrics:** Railway dashboard → Metrics tab
- **Deployments:** Railway dashboard → Deployments tab

## Cost

Railway offers:
- Free tier: $5 credit/month
- Hobby plan: $20/month (includes more resources)
- Pay-as-you-go for additional usage

Check Railway pricing for current rates.

