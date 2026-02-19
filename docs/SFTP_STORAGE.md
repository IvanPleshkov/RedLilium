# SFTP Remote Storage Setup

This guide explains how to configure an Ubuntu server as remote asset storage for the RedLilium Editor via SFTP.

## Prerequisites

- Ubuntu server with a static IP address
- SSH access to the server (root or sudo)
- An SSH key pair on your development machine (e.g. `~/.ssh/hetzner-amd`)

## 1. Install OpenSSH Server

Most Ubuntu servers have this already. If not:

```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

## 2. Create a Dedicated User

Create a user for the editor to connect as:

```bash
sudo adduser deploy --disabled-password
```

If you already have a user you want to use, skip this step.

## 3. Create the Storage Directory

```bash
sudo mkdir -p /data/assets
sudo chown deploy:deploy /data/assets
```

Choose any path you like — this will be the `path` in `project.toml`.

## 4. Deploy Your SSH Public Key

On your **development machine**, copy the public key to the server:

```bash
ssh-copy-id -i ~/.ssh/hetzner-amd.pub deploy@YOUR_SERVER_IP
```

Or manually:

```bash
# On your dev machine, print the public key:
cat ~/.ssh/hetzner-amd.pub

# On the server, as root or sudo:
sudo mkdir -p /home/deploy/.ssh
sudo nano /home/deploy/.ssh/authorized_keys
# Paste the public key, save

sudo chmod 700 /home/deploy/.ssh
sudo chmod 600 /home/deploy/.ssh/authorized_keys
sudo chown -R deploy:deploy /home/deploy/.ssh
```

## 5. Test the Connection

From your development machine:

```bash
ssh -i ~/.ssh/hetzner-amd deploy@YOUR_SERVER_IP
```

You should log in without a password prompt. Then verify SFTP works:

```bash
sftp -i ~/.ssh/hetzner-amd deploy@YOUR_SERVER_IP
sftp> ls /data/assets
sftp> exit
```

## 6. Configure project.toml

In your project's `project.toml`, add an SFTP mount:

```toml
[[mount]]
name = "remote"
type = "sftp"
host = "YOUR_SERVER_IP"
port = 22
username = "deploy"
key = "~/.ssh/hetzner-amd"
path = "/data/assets"
```

The editor will connect on startup and show the remote files in the asset browser.

## 7. Optional: Restrict User to SFTP Only

For security, you can restrict the `deploy` user to SFTP access only (no shell login):

```bash
sudo nano /etc/ssh/sshd_config
```

Add at the end of the file:

```
Match User deploy
    ForceCommand internal-sftp
    ChrootDirectory /data
    AllowTcpForwarding no
    X11Forwarding no
```

If using `ChrootDirectory`, the chroot path (`/data`) must be owned by root:

```bash
sudo chown root:root /data
sudo chown deploy:deploy /data/assets
```

The `path` in `project.toml` becomes relative to the chroot — change it to `/assets`:

```toml
path = "/assets"
```

Restart SSH to apply:

```bash
sudo systemctl restart ssh
```

## Troubleshooting

**Connection refused** — Check that SSH is running (`sudo systemctl status ssh`) and the firewall allows port 22 (`sudo ufw allow 22`).

**Permission denied (publickey)** — Verify the public key is in `/home/deploy/.ssh/authorized_keys` and file permissions are correct (700 for `.ssh`, 600 for `authorized_keys`).

**Cannot read/write files** — Check that the `deploy` user owns the storage directory (`ls -la /data/assets`).

**Editor shows "Failed to connect SFTP mount"** — Check the log output for the specific SSH/SFTP error. Common causes: wrong IP, wrong key path, wrong username, or SSH key not accepted.
