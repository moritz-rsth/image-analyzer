# Gunicorn configuration for Google Cloud Compute Engine
# TODO: Adjust workers based on VM instance size
# TODO: Configure logging paths
# TODO: Set up proper error handling

import multiprocessing
import os

# Bind to all interfaces on the port specified by environment variable
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# Worker configuration
# Adjust based on your VM instance size
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "eventlet"
worker_connections = 1000

# Timeout settings
timeout = 120
keepalive = 5

# Request limits
max_requests = 1000
max_requests_jitter = 50

# Logging (TODO: configure proper log paths)
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# Process naming
proc_name = "image-analyzer"

