# Platform Packages Installation Guide

This guide covers installing ProximaDB from native platform packages on Linux and Windows. Each
release on [GitHub Releases](https://github.com/anvai-labs/proximaDB/releases) ships RPM, DEB and
MSI packages. The commands resolve the latest release:

```bash
VERSION=$(curl -fsSL https://api.github.com/repos/anvai-labs/proximaDB/releases/latest \
  | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p')
BASE=https://github.com/anvai-labs/proximaDB/releases/download/v$VERSION
```

> **Known issues:** the RPM/DEB service unit cannot start and the packaged config keeps data
> under `/tmp` (TD-PKG-1); the MSI installs no Windows service (TD-PKG-2). Both are scheduled for
> v0.5. Until then, run the server binary directly or use Docker; see the
> [installation guide](../01-quick-start/install.md) for workarounds.

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

See [Linux requirements](../01-quick-start/install.md#linux-requirements-archive-deb-rpm-homebrew)
for the distributions each release is verified on. Note that **v0.4.1 and earlier** need glibc
2.35 and `libssl.so.3`: their `.el8.rpm` does not run on RHEL/Rocky 8 or 9.

### Installation

```bash
# Download the RPM package
wget "$BASE/proximadb-$VERSION-1.el8.x86_64.rpm"

# Install the package
sudo rpm -ivh "proximadb-$VERSION-1.el8.x86_64.rpm"

# Start the service and enable it on boot (fails until the TD-PKG-1 workaround;
# see the note at the top of this guide)
sudo systemctl start proximadb
sudo systemctl enable proximadb
```

---

## Linux DEB Package (Debian/Ubuntu)

### Supported Distributions

See [Linux requirements](../01-quick-start/install.md#linux-requirements-archive-deb-rpm-homebrew).
**v0.4.1 and earlier** need Ubuntu 22.04+ or Debian 12+.

### Installation

```bash
# Download and install the DEB package, then start/enable the service
# exactly as for the RPM above (sudo systemctl start/enable proximadb)
wget "$BASE/proximadb_${VERSION}_amd64.deb"
sudo dpkg -i "proximadb_${VERSION}_amd64.deb"
```

Both Linux packages install the same file layout and are verified the same way — see
[Post-Installation](#post-installation) below for what gets installed, verification and
uninstallation (only the package-manager command differs: `rpm` vs `dpkg`).

---

## Windows MSI Installer

### Supported Versions

- Windows 10 (64-bit)
- Windows 11 (64-bit)
- Windows Server 2019+

### Installation

**Method 1: Double-click**
1. Download `proximadb-<version>-x64.msi`
2. Double-click the file
3. Follow the installation wizard
4. Complete the installation

**Method 2: Command Line**
```powershell
# Download the MSI (or use the link above)
msiexec /i proximadb-<version>-x64.msi
```

**Method 3: Silent Installation**
```powershell
msiexec /i proximadb-<version>-x64.msi /qn /norestart
```

### What Gets Installed

- **Program Files**: `C:\Program Files\ProximaDB\`
  - `proximadb-server.exe`
  - `proximadb-bench.exe`
  - `proximadb-migrate.exe`
- **Configuration**: `C:\Program Files\ProximaDB\config.toml`
- **Data**: `C:\ProgramData\ProximaDBData\` is created empty; the shipped config writes under `\tmp\proximadb\` until you point its roots there (TD-PKG-2)
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
msiexec /x proximadb-<version>-x64.msi
```

---

## Post-Installation

### What Gets Installed (Linux packages)

The RPM and DEB packages install the same layout:

- **Binaries**: `/usr/bin/` (proximadb-server, proximadb-bench, proximadb-migrate)
- **Configuration**: `/etc/proximadb/config.toml`
- **Service**: `/usr/lib/systemd/system/proximadb.service` (`/lib/systemd/system/` on Debian — the same file)
- **Data**: `/var/lib/proximadb/` (after the TD-PKG-1 workaround in the installation guide; the shipped config uses `/tmp`)
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

### Uninstallation (Linux packages)

```bash
# Stop and disable the service
sudo systemctl stop proximadb
sudo systemctl disable proximadb

# Remove the package (RPM on the left, DEB on the right)
sudo rpm -e proximadb        # sudo dpkg -r proximadb

# Optionally remove data and logs
sudo rm -rf /var/lib/proximadb /var/log/proximadb
```

### Default Configuration

The default configuration file is installed at:

- **Linux**: `/etc/proximadb/config.toml`
- **Windows**: `C:\Program Files\ProximaDB\config.toml`

### Default Ports

- **Unified Port**: `5678` (REST, gRPC; Arrow Flight experimental)
- **PostgreSQL Wire Protocol**: `5433` (enabled by default, bound to `127.0.0.1`)

### Default Data Directory

- **Linux**: `/var/lib/proximadb`
- **Windows**: `C:\ProgramData\ProximaDBData`

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

If installed as a service (the MSI does not install one yet — TD-PKG-2; these commands apply only
to a service you registered yourself, for example with `sc.exe create`):

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
notepad "C:\Program Files\ProximaDB\config.toml"
```

### Key Configuration Options

```toml
[server]
# Port for unified REST/gRPC/Arrow Flight (default: 5678)
port = 5678

# PostgreSQL wire protocol port (default: 5433, bound to 127.0.0.1).
#
# Leave this COMMENTED unless the pgwire listener authenticates. Setting it makes
# the server re-derive the pgwire bind from `[server] bind_address`; if that is
# not loopback, the boot-time listener preflight refuses an unauthenticated
# single-tenant pgwire listener and the server exits 1. Enable pgwire SCRAM first
# (`[security.authentication] enabled = true`, or `[security.pgwire] auth =
# "password"`) -- TD-PGWIRE-AUTH-1 / TD-PGWIRE-CONTAINER-1.
# pg_port = 5433

[storage]
# The vector engine is chosen per collection at create time (pass "engine": "sst"); there is no server-wide key.

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

# Run in the foreground to see configuration errors
sudo -u proximadb proximadb-server --config /etc/proximadb/config.toml
```

**Windows:**
```powershell
# No Windows service is installed (TD-PKG-2); run the server in the foreground instead

# Run server manually to see errors
& "C:\Program Files\ProximaDB\proximadb-server.exe" --config "C:\Program Files\ProximaDB\config.toml"
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

# Run in the foreground to see configuration errors
proximadb-server --config /etc/proximadb/config.toml

# Health check
curl http://localhost:5678/health

# Metrics
curl http://localhost:5678/metrics
```

---

## Next Steps

After installation:

1. **Configure**: Edit `/etc/proximadb/config.toml` (Linux) or `C:\Program Files\ProximaDB\config.toml` (Windows)
2. **Start**: `sudo systemctl start proximadb` (Linux, once TD-PKG-1 is fixed) or run `proximadb-server.exe` (Windows; no service until TD-PKG-2)
3. **Verify**: `curl http://localhost:5678/health`
4. **Create Collection**: Use the API or Python SDK
5. **Start Ingesting**: Add vectors and data

For more information, see the [API Documentation](../03-api-reference/index.md).
