# Google Cloud Platform Deployment Guide

This guide explains how to deploy the Image Analyzer to Google Cloud Compute Engine.

## Prerequisites

1. Google Cloud Platform account
2. Billing enabled
3. Google Cloud SDK installed (optional, for local setup)
4. VM instance created

## Setup Steps

### 1. Create VM Instance

1. Go to Google Cloud Console → Compute Engine → VM Instances
2. Click "Create Instance"
3. Configure:
   - **Name:** `image-analyzer-vm`
   - **Machine type:** `e2-standard-4` (or `e2-standard-2` for lower cost)
   - **Boot disk:** Ubuntu 22.04 LTS
   - **Firewall:** Allow HTTP and HTTPS traffic (or create custom rule for port 5000)
4. Click "Create"

### 2. Reserve Static IP (Recommended)

1. Go to VPC network → External IP addresses
2. Click "Reserve static address"
3. Name: `image-analyzer-ip`
4. Attach to your VM instance

### 3. Configure Firewall Rule

1. Go to VPC network → Firewall
2. Click "Create Firewall Rule"
3. Configure:
   - **Name:** `allow-image-analyzer`
   - **Direction:** Ingress
   - **Action:** Allow
   - **Targets:** All instances in the network
   - **Source IP ranges:** `0.0.0.0/0` (or restrict to specific IPs)
   - **Protocols and ports:** TCP port `5000`
4. Click "Create"

### 4. SSH into VM

```bash
gcloud compute ssh image-analyzer-vm
# Or use SSH button in Google Cloud Console
```

### 5. Install Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.12 and pip
sudo apt-get install -y python3.12 python3.12-venv python3-pip

# Install system dependencies for image processing
sudo apt-get install -y tesseract-ocr libtesseract-dev libopencv-dev

# Clone your repository (or upload files)
# Option 1: Clone from GitHub
git clone <your-repo-url>
cd image-analyzer

# Option 2: Upload files via SCP
# scp -r . user@VM_IP:/path/to/image-analyzer
```

### 6. Set Up Application

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirementsDeployment.txt
```

### 7. Configure Environment Variables

Create `.env` file:

```bash
nano .env
```

Add:

```bash
REPLICATE_API_TOKEN=your-replicate-token-here
SECRET_KEY=your-secret-key-here
ADMIN_PASSWORD_HASH=your-sha256-hash-here
APP_BASE_URL=http://YOUR_STATIC_IP:5000  # REQUIRED: Replace with your VM's external IP
PORT=5000
FLASK_ENV=production
```

**Important:**
- `APP_BASE_URL` is **required** on GCP - set it to your VM's external IP address
- The app will auto-detect GCP but cannot determine your IP automatically

**Generate values:**
```bash
# SECRET_KEY
python3 -c "import secrets; print(secrets.token_hex(32))"

# ADMIN_PASSWORD_HASH
python3 -c "import hashlib; print(hashlib.sha256(b'your-password').hexdigest())"
```

### 8. Configure Gunicorn

The Gunicorn config is in `deployments/gcp/gunicorn_config.py`. Adjust workers based on your VM size:

- `e2-standard-2`: 5 workers
- `e2-standard-4`: 9 workers
- `e2-standard-8`: 17 workers

### 9. Create Systemd Service (Recommended)

Create service file:

```bash
sudo nano /etc/systemd/system/image-analyzer.service
```

Add:

```ini
[Unit]
Description=Image Analyzer Flask App
After=network.target

[Service]
User=YOUR_USERNAME
WorkingDirectory=/path/to/image-analyzer
Environment="PATH=/path/to/image-analyzer/venv/bin"
ExecStart=/path/to/image-analyzer/venv/bin/gunicorn -c deployments/gcp/gunicorn_config.py app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable image-analyzer
sudo systemctl start image-analyzer
sudo systemctl status image-analyzer
```

### 10. Alternative: Use Startup Script

You can also use the startup script (`deployments/gcp/startup-script.sh`):

1. Edit the script with your actual paths
2. Upload to Google Cloud Storage
3. Configure VM to use it as startup script

## Access Your Application

Your app will be available at:
```
http://YOUR_STATIC_IP:5000
```

## HTTPS Setup (Optional)

For HTTPS, you have several options:

### Option 1: Google Cloud Load Balancer
- Create Load Balancer with SSL certificate
- Point to your VM instance
- Cost: ~$18/month + traffic

### Option 2: Let's Encrypt with Nginx
- Install Nginx as reverse proxy
- Use Certbot for SSL certificates
- Free, but requires domain name

## Monitoring

- **Logs:** `sudo journalctl -u image-analyzer -f`
- **VM Metrics:** Google Cloud Console → Compute Engine → VM Instances → Your VM → Monitoring
- **Application Logs:** Check Gunicorn logs (configured in `gunicorn_config.py`)

## Troubleshooting

### App Not Accessible
- Check firewall rule allows port 5000
- Verify VM has external IP
- Check Gunicorn is running: `sudo systemctl status image-analyzer`

### High Memory Usage
- Reduce workers in `gunicorn_config.py`
- Consider upgrading VM instance

### Replicate API Errors
- **Verify `APP_BASE_URL` is set to your VM's external IP** (e.g., `http://34.123.45.67:5000`)
- Ensure firewall allows inbound connections on port 5000
- Check Replicate API token is valid

## Cost Estimation

- **e2-standard-4 VM:** ~$100-150/month (24/7)
- **e2-standard-2 VM:** ~$50-75/month
- **Static IP:** Free (if VM is running)
- **Traffic:** ~$0.12/GB egress

## Security Recommendations

1. **Restrict Firewall:** Only allow specific IP ranges if possible
2. **Use Strong Passwords:** Generate secure `SECRET_KEY` and `ADMIN_PASSWORD_HASH`
3. **Regular Updates:** Keep system and dependencies updated
4. **Backup:** Regularly backup `uploads/` and `outputs/` folders
5. **HTTPS:** Consider setting up HTTPS for production use

## Next Steps

- Set up automated backups
- Configure monitoring alerts
- Set up HTTPS (if needed)
- Configure custom domain (optional)

