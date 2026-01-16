# Railway Deployment Guide

This guide explains how to deploy the Image Analyzer to Railway using Docker.

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
5. Railway will auto-detect the `Dockerfile` in the root directory

### 2. Configure Builder

In Railway Settings:
1. Go to **Settings → Build**
2. **Builder**: Select "Docker" (not Railpack or Nixpacks)
3. **Custom Build Command**: Leave empty
4. Railway will automatically use the `Dockerfile` in the root directory

### 3. Configure Environment Variables

In Railway dashboard, go to your project → Variables tab and add:

```bash
REPLICATE_API_TOKEN=your-replicate-token-here
SECRET_KEY=your-secret-key-here  # Generate with: python3 -c "import secrets; print(secrets.token_hex(32))"
ADMIN_PASSWORD_HASH=your-sha256-hash-here  # Generate with: python3 -c "import hashlib; print(hashlib.sha256(b'your-password').hexdigest())"
```

**Important:** 
- Railway automatically sets `RAILWAY_ENVIRONMENT`, `PORT`, and `RAILWAY_PUBLIC_DOMAIN`
- The app will auto-detect Railway and automatically set `APP_BASE_URL` to `https://RAILWAY_PUBLIC_DOMAIN`
- **You don't need to set `APP_BASE_URL` or `PORT` manually on Railway** - they're auto-configured!

### 4. Deploy

1. Railway will automatically deploy on every push to main branch
2. Or click "Deploy" in Railway dashboard
3. Wait for Docker build to complete
4. Your app will be available at: `https://image-analyzer.up.railway.app` (or your custom domain)

## Docker Configuration

The project includes a `Dockerfile` in the root directory that:
- Uses Python 3.12-slim base image
- Installs system dependencies (OpenCV, Tesseract OCR)
- Installs Python packages from `requirementsDeployment.txt`
- Starts the app with Gunicorn + Eventlet worker

The Dockerfile is optimized for Railway:
- Multi-stage caching for faster builds
- Minimal base image
- Proper port configuration (`${PORT:-5000}`)

## Railway-Specific Features

### Automatic HTTPS
Railway provides HTTPS automatically via their domain. No SSL certificate setup needed.

### Port Configuration
Railway automatically sets the `PORT` environment variable. The Dockerfile listens on `0.0.0.0:$PORT`.

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
- Check that `Dockerfile` is in the root directory
- Verify that `requirementsDeployment.txt` exists
- Check build logs in Railway dashboard
- Ensure Docker builder is selected (not Railpack/Nixpacks)

### App Crashes on Startup
- Verify all environment variables are set
- Check logs: Railway dashboard → Deployments → View Logs
- Ensure `SECRET_KEY` and `ADMIN_PASSWORD_HASH` are set
- Check that port is correctly configured (Railway sets `PORT` automatically)

### Replicate API Errors
- Verify `REPLICATE_API_TOKEN` is set correctly
- Check that `APP_BASE_URL` is set correctly (auto-set on Railway, check logs if issues)
- Replicate requires publicly accessible URLs (Railway provides HTTPS automatically)

### SocketIO Not Working
- Ensure `eventlet` is installed (already in requirementsDeployment.txt)
- Check that Gunicorn uses `eventlet` worker class (configured in Dockerfile CMD)

### Docker Build Issues
- Check `.dockerignore` to ensure necessary files are included
- Verify system dependencies are installed (OpenCV, Tesseract)
- Check Docker build logs for specific errors

## Local Docker Testing

Before deploying to Railway, you can test locally:

```bash
# Build the image
docker build -t image-analyzer .

# Run the container
docker run -p 5000:5000 \
  -e SECRET_KEY=test-secret \
  -e ADMIN_PASSWORD_HASH=<hash> \
  -e REPLICATE_API_TOKEN=<token> \
  -e PORT=5000 \
  image-analyzer
```

## Monitoring

- **Logs:** Railway dashboard → Deployments → View Logs
- **Metrics:** Railway dashboard → Metrics tab
- **Deployments:** Railway dashboard → Deployments tab

