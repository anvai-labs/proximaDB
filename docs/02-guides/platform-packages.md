# Platform Packages Installation Guide

This guide covers installing ProximaDB v0.2.0 using native platform packages on Linux and Windows.

## Table of Contents

- [Linux RPM Package (RHEL/CentOS/Fedora)](#linux-rpm-package-rhelcentosfedora)
- [Linux DEB Package (Debian/Ubuntu)](#linux-deb-package-debianubuntu)
- [Windows MSI Installer](#windows-msi-installer)
- [Post-Installation](#post-installation)
- [Service Management](#service-management)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Linux RPM Package (RHEL/CentOS/Fedora)

### Supported Distributions

- Red Hat Enterprise Linux (RHEL) 8+
- CentOS 8+
- Fedora 35+
- Other RPM-based distributions

### Installation

```bash
# Download the RPM package
wget https://github.com/anvai-labs/proximaDB/releases/download/v0.2.0/proximadb-0.2.0-1.el8.x86_64.rpm

# Install the package
sudo rpm -ivh proximadb-0.2.0-1.el8.x86_64.rpm

# Start the service
sudo systemctl start proximadb

# Enable the service to start on boot
sudo systemctl enable proximadb
```

### What Gets Installed

- **Binaries**: `/usr/bin/` (proximadb-server, proximadb-bench, proximadb-migrate)
- **Configuration**: `/etc/proximadb/config.toml`
- **Service**: `/usr/lib/systemd/system/proximadb.service`
- **Data**: `/var/lib/proximadb/`
- **Logs**: `/var/log/proximadb/`

### Verification

```bash
# Check service status
sudo systemctl status proximadb

# Check version
proximadb-server --version

# Check if server is running
curl http://localhost:5678/health
```

### Uninstallation

```bash
# Stop and disable the service
sudo systemctl stop proximadb
sudo systemctl disable proximadb

# Remove the package
sudo rpm -e proximadb

# Optionally remove data and logs
sudo rm -rf /var/lib/proximadb /var/log/proximadb
```

---

## Linux DEB Package (Debian/Ubuntu)

### Supported Distributions

- Debian 11+
- Ubuntu 20.04+
- Other DEB-based distributions

### Installation

```bash
# Download the DEB package
wget https://github.com/anvai-labs/proximaDB/releases/download/v0.2.0/proximadb_0.2.0_amd64.deb

# Install the package
sudo dpkg -i proximadb_0.2.0_amd64.deb

# Start the service
sudo systemctl start proximadb

# Enable the service to start on boot
sudo systemctl enable proximadb
```

### What Gets Installed

- **Binaries**: `/usr/bin/` (proximadb-server, proximadb-bench, proximadb-migrate)
- **Configuration**: `/etc/proximadb/config.toml`
- **Service**: `/lib/systemd/system/proximadb.service`
- **Data**: `/var/lib/proximadb/`
- **Logs**: `/var/log/proximadb/`

### Verification

```bash
# Check service status
sudo systemctl status proximadb

# Check version
proximadb-server --version

# Check if server is running
curl http://localhost:5678/health
```

### Uninstallation

```bash
# Stop and disable the service
sudo systemctl stop proximadb
sudo systemctl disable proximadb

# Remove the package
sudo dpkg -r proximadb

# Optionally remove data and logs
sudo rm -rf /var/lib/proximadb /var/log/proximadb
```

---

## Windows MSI Installer

### Supported Versions

- Windows 10 (64-bit)
- Windows 11 (64-bit)
- Windows Server 2019+

### Installation

**Method 1: Double-click**
1. Download `proximadb-0.2.0-x64.msi`
2. Double-click the file
3. Follow the installation wizard
4. Complete the installation

**Method 2: Command Line**
```powershell
# Download the MSI (or use the link above)
msiexec /i proximadb-0.2.0-x64.msi
```

**Method 3: Silent Installation**
```powershell
msiexec /i proximadb-0.2.0-x64.msi /qn /norestart
```

### What Gets Installed

- **Program Files**: `C:\Program Files\ProximaDB\`
  - `proximadb-server.exe`
  - `proximadb-bench.exe`
  - `proximadb-migrate.exe`
- **Configuration**: `C:\ProgramData\ProximaDB\config.toml`
- **Data**: `C:\ProgramData\ProximaDB\data\`
- **Service**: Windows Service (optional)

### Verification

```powershell
# Check version
& "C:\Program Files\ProximaDB\proximadb-server.exe" --version

# Check if server is running
curl http://localhost:5678/health
```

### Uninstallation

**Method 1: Control Panel**
1. Go to Settings > Apps
2. Find "ProximaDB"
3. Click "Uninstall"

**Method 2: Command Line**
```powershell
msiexec /x proximadb-0.2.0-x64.msi
```

---

## Post-Installation

### Default Configuration

The default configuration file is installed at:

- **Linux**: `/etc/proximadb/config.toml`
- **Windows**: `C:\ProgramData\ProximaDB\config.toml`

### Default Ports

- **Unified Port**: `5678` (REST, gRPC; Arrow Flight experimental)
- **PostgreSQL Wire Protocol**: `5433` (if enabled)

### Default Data Directory

- **Linux**: `/var/lib/proximadb`
- **Windows**: `C:\ProgramData\ProximaDB\data`

---

## Service Management

### Linux (systemd)

```bash
# Start service
sudo systemctl start proximadb

# Stop service
sudo systemctl stop proximadb

# Restart service
sudo systemctl restart proximadb

# Check status
sudo systemctl status proximadb

# View logs
sudo journalctl -u proximadb -f

# Enable on boot
sudo systemctl enable proximadb

# Disable from boot
sudo systemctl disable proximadb
```

### Windows Service

If installed as a service:

```powershell
# Start service
Start-Service proximadb

# Stop service
Stop-Service proximadb

# Restart service
Restart-Service proximadb

# Check status
Get-Service proximadb

# View event logs
Get-EventLog -LogName Application -Source proximadb -Newest 100
```

---

## Configuration

### Basic Configuration

Edit the configuration file:

**Linux:**
```bash
sudo nano /etc/proximadb/config.toml
```

**Windows:**
```powershell
notepad "C:\ProgramData\ProximaDB\config.toml"
```

### Key Configuration Options

```toml
[server]
# Port for unified REST/gRPC/Arrow Flight (default: 5678)
port = 5678

# PostgreSQL wire protocol port (default: 5433)
# pg_port = 5433

[storage]
# Storage engine: sst, helix, viper, nova, raptor, swift
default_engine = "sst"

[api]
# API settings
unified_mode = true
```

### Apply Configuration Changes

```bash
# Restart the service to apply changes
sudo systemctl restart proximadb
```

---

## Troubleshooting

### Service Won't Start

**Linux:**
```bash
# Check service status
sudo systemctl status proximadb

# View logs
sudo journalctl -u proximadb -n 50

# Check configuration
sudo proximadb-server --config /etc/proximadb/config.toml --check
```

**Windows:**
```powershell
# Check service status
Get-Service proximadb

# Run server manually to see errors
& "C:\Program Files\ProximaDB\proximadb-server.exe" --config "C:\ProgramData\ProximaDB\config.toml"
```

### Port Already in Use

```bash
# Check what's using the port
sudo lsof -i :5678

# Kill the process if needed
sudo kill -9 <PID>
```

### Permission Errors

**Linux:**
```bash
# Check file permissions
ls -la /var/lib/proximadb

# Fix permissions if needed
sudo chown -R proximadb:proximadb /var/lib/proximadb
sudo chmod -R 755 /var/lib/proximadb
```

### Can't Connect to Server

```bash
# Check if server is running
sudo systemctl status proximadb

# Check firewall
sudo firewall-cmd --list-ports
sudo firewall-cmd --add-port=5678/tcp --permanent
sudo firewall-cmd --reload

# Test locally
curl http://localhost:5678/health
```

### Windows: Server Won't Start

1. Check Event Viewer for errors
2. Verify configuration file syntax
3. Run as Administrator if needed
4. Check antivirus software isn't blocking the executable

---

## Advanced Configuration

### Custom Data Directory

**Linux:**
```bash
# Edit service file
sudo systemctl edit proximadb

# Add custom data directory
EnvironmentDataDirectory=/custom/path/to/data
```

**Windows:**
```powershell
# Modify service parameters
sc config proximadb binPath= "C:\Program Files\ProximaDB\proximadb-server.exe --config C:\Custom\config.toml"
```

### Running as Different User

**Linux:**
```bash
# Create service override
sudo systemctl edit proximadb

# Add user
[Service]
User=myuser
Group=mygroup
```

### Resource Limits

**Linux:**
```bash
# Create override directory
sudo systemctl edit proximadb

# Add resource limits
[Service]
LimitNOFILE=65536
MemoryMax=2G
```

---

## Getting Help

### Documentation

- [Configuration Guide](../03-api-reference/configuration.adoc)
- [API Documentation](../03-api-reference/index.md)
- [Architecture](../05-concepts/architecture.adoc)

### Community

- GitHub Issues: https://github.com/anvai-labs/proximaDB/issues
- Discussions: https://github.com/anvai-labs/proximaDB/discussions

### Debug Information

```bash
# Get server version
proximadb-server --version

# Get build information
proximadb-server --build-info

# Check configuration
proximadb-server --config /etc/proximadb/config.toml --check

# Health check
curl http://localhost:5678/health

# Metrics
curl http://localhost:5678/metrics
```

---

## Next Steps

After installation:

1. **Configure**: Edit `/etc/proximadb/config.toml` (Linux) or `C:\ProgramData\ProximaDB\config.toml` (Windows)
2. **Start Service**: `sudo systemctl start proximadb` (Linux) or Start Service (Windows)
3. **Verify**: `curl http://localhost:5678/health`
4. **Create Collection**: Use the API or Python SDK
5. **Start Ingesting**: Add vectors and data

For more information, see the [API Documentation](../03-api-reference/index.md).
