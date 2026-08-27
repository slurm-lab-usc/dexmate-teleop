# Dexmate Vega U Workstation Setup (Multi-User)

This guide explains how to set up a workstation so multiple Linux accounts can
connect to the same Dexmate Vega U robot. The primary workflow is exoskeleton
teleoperation.

## Robot Information

The validated robot used by this guide:

```text
Hostname:                vega-1u
Robot name:              dm/vge07dbe2d05-1u
Robot public/external IP: 192.168.50.20
Robot SSH username:       dexmate
Robot SSH password:       hello-dex
Internal management UI:   http://192.168.50.21:57832
Management UI password:   slurm@dexmate
```

Replace these values if you are configuring a different robot.

`192.168.50.20` is the robot's public/external interface on the private robot
network and supports SSH access. The management UI on `192.168.50.21:57832`
is an internal web service, not an SSH endpoint, and uses a separate password.
Administrators keep the management UI recovery code separately in a private
repository; the recovery code must not be committed to this repository.

## Configuration Overview

For a shared workstation, configuration is split into two levels:

| Level | What is included | Who is responsible |
|-------|------------------|--------------------|
| Admin / machine-level | Ethernet/NetworkManager, robot-side relay and camera services, internal management UI, certificate renewal, shared certificate directory | Administrator |
| User-level | Personal conda environment, environment variables, access to the shared certificate | Each lab member |

On the current workstation, the admin/machine-level configuration is already
in place. New lab members normally only need the user-level section.

---

# A. Admin / Machine-Level Configuration

This section is for initial machine setup or troubleshooting. It is usually
not needed when adding a new user.

## A.1 Network Layout

The workstation and robot use fixed addresses on a private Ethernet subnet:

```text
Workstation Ethernet:      192.168.50.10/24
Robot public/external IP:  192.168.50.20/24
Robot internal UI:         192.168.50.21:57832
Subnet mask:               255.255.255.0
Gateway:                   none
DNS:                       none
```

Notes:

- The workstation address belongs to the workstation's Ethernet interface, not
  to the robot.
- Do not assign `.20` or `.21` to the workstation. `.20` is the robot's
  public/external communication interface, and `.21` is reserved for the
  internal management UI.
- In `192.168.50.21:57832`, `.21` is part of the IP address and `57832` is the
  web service's TCP port.
- Mark the robot connection as `never-default` so it does not replace the
  normal Internet route.

## A.2 Configure the Workstation Ethernet (New Machines Only)

Find the robot-facing interface:

```bash
ip -brief address
nmcli device status
```

On the validated machine, the interface is `enp10s0`. Create a persistent
NetworkManager connection:

```bash
sudo nmcli connection add \
  type ethernet \
  ifname enp10s0 \
  con-name Dexmate-VegaU \
  ipv4.method manual \
  ipv4.addresses 192.168.50.10/24 \
  ipv4.never-default yes \
  ipv6.method disabled \
  connection.autoconnect yes
```

If it does not activate automatically:

```bash
sudo nmcli connection up Dexmate-VegaU
```

Verify:

```bash
ip -brief address show dev enp10s0
ip route get 192.168.50.20
ping -c 4 192.168.50.20
```

## A.3 Robot-Side Services

The robot must run two services:

- `dextop-node.service`: Zenoh TLS relay, listening on `7447`
- `dexsensor.service`: head camera publisher

Check status:

```bash
ssh dexmate@192.168.50.20
systemctl status dextop-node.service --no-pager
systemctl status dexsensor.service --no-pager
dexsensor status --robot dm/vge07dbe2d05-1u head_camera
```

If a service is not running:

```bash
sudo systemctl restart dextop-node.service
sudo systemctl restart dexsensor.service
```

From the workstation, confirm the relay is reachable:

```bash
timeout 5 bash -c '</dev/tcp/192.168.50.20/7447'
```

No output and a zero exit status means the connection succeeded.

### A.3.1 Log in to the Robot (New Machine Setup Reference)

```bash
ssh dexmate@192.168.50.20
```

Log in with the robot SSH password listed in
[Robot Information](#robot-information). The internal management UI uses a
separate password; do not try that password at the SSH prompt.

Confirm identity and time:

```bash
hostname
echo "$ROBOT_NAME"
date
```

### A.3.2 Robot Conda Environment

Dexmate tools on the robot live in the `dexmate_env1` environment:

```bash
conda activate dexmate_env1
python --version
dextop --version
python -c "import importlib.metadata as m; print(m.version('dexcontrol'))"
```

Expected versions:

```text
Python 3.10
dextop 0.5.0
dexcontrol 0.5.0
```

Do not run a general `apt upgrade` or upgrade Dexmate packages without checking
firmware and Jetson compatibility.

### A.3.3 Start the Relay Manually (Debugging)

```bash
conda activate dexmate_env1
dextop node start
```

Keep the command running. You can use tmux:

```bash
tmux new -s relay
conda activate dexmate_env1
dextop node start
```

Detach with `Ctrl+B`, then `D`. Reattach with `tmux attach -t relay`.

### A.3.4 Install the Relay as a systemd Service (New Machine Setup Reference)

```bash
cat > /home/dexmate/start-dextop-node.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /home/dexmate/miniconda3/etc/profile.d/conda.sh
conda activate dexmate_env1

export ROBOT_NAME=dm/vge07dbe2d05-1u
export ZENOH_CONFIG=/home/dexmate/.dexmate/comm/zenoh/VGE07DBE2D05.dzcfg

exec dextop node start
EOF

chmod +x /home/dexmate/start-dextop-node.sh

sudo tee /etc/systemd/system/dextop-node.service >/dev/null <<'EOF'
[Unit]
Description=DexTop Zenoh relay node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=dexmate
WorkingDirectory=/home/dexmate
ExecStart=/home/dexmate/start-dextop-node.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now dextop-node.service
sudo systemctl status dextop-node.service --no-pager
```

### A.3.5 Install the Head Camera Service (New Machine Setup Reference)

```bash
sudo cp /etc/dexmate/dexsensor/default.toml /etc/dexmate/dexsensor/head_camera.toml

sudo python3 - <<'PY'
from pathlib import Path

p = Path("/etc/dexmate/dexsensor/head_camera.toml")
text = p.read_text()
text = text.replace('name = "default"', 'name = "dm/vge07dbe2d05-1u"', 1)

marker = 'id = "head_camera"'
start = text.index(marker)
enabled = text.index("enabled = false", start)
text = text[:enabled] + "enabled = true" + text[enabled + len("enabled = false"):]

p.write_text(text)
PY

pkill -f 'dexsensor.*head_camera' || true

sudo dexsensor install --config /etc/dexmate/dexsensor/head_camera.toml
sudo systemctl enable --now dexsensor.service
sudo systemctl status dexsensor.service --no-pager
```

### A.3.6 Check Robot Health (Read-Only)

On the robot:

```bash
dextop doctor check
dextop topic list --timeout 10
dextop firmware info
```

A healthy result should show:

- MCU connectivity passing
- robot server running
- Zenoh communication configuration passing
- time synchronization passing
- robot state topics available

A Socbridge diagnostics warning may still appear while the device is reachable
through an SSH fallback. Record the warning, but do not update firmware unless
required by the official procedure.

## A.4 Internal Management UI

Administrators can open the robot's internal management UI from a workstation
on the robot network:

```text
http://192.168.50.21:57832
```

Log in with the management UI password listed in
[Robot Information](#robot-information). This interface is used for
administrator-only maintenance, including uploading the robot certificate
package. The recovery code is stored separately in an administrator-only
private repository and is shared only with other robot administrators.

## A.5 Certificate Renewal and Shared Certificate Directory

The robot certificate expires approximately every 90 days. An expired
certificate can prevent robot communication and operation even when
`192.168.50.20` still replies to `ping`. Administrators should track the
expiration date and renew the certificate before it expires.

### A.5.1 Renew the Robot Certificate

Certificate renewal uses the robot's internal management UI and the
`dextop`-provided package extraction/import mechanism:

1. Obtain the renewed certificate package from the approved source.
2. Open `http://192.168.50.21:57832` and log in as an administrator.
3. Open the certificate management page and upload the renewed package.
4. Complete the `dextop`-managed extraction/import flow shown by the UI or
   supplied with the certificate package.
5. Restart services or reboot only if the Dexmate renewal procedure requests
   it.
6. From the workstation, run `dextop topic list --timeout 20` and confirm that
   the robot topics are available.

Do not unpack the certificate package with a generic archive tool or manually
replace robot-side certificate files. The exact UI labels and `dextop` command
may change between Dexmate releases, so administrators must follow the renewal
instructions supplied with the current package.

### A.5.2 Share the Resulting Workstation Certificate

The `.dzcfg` file is a sensitive robot access credential. Do not commit it to
Git, upload it to an issue, or share it through an unapproved channel.

The recommended setup is one shared certificate on the workstation, with all
lab members pointing to it through `ZENOH_CONFIG`.

Administrator setup:

```bash
# 1. Create a dedicated group, e.g. dexlab
sudo groupadd dexlab || true

# 2. Create the shared certificate directory
sudo mkdir -p /srv/dexmate/certs
sudo chown root:dexlab /srv/dexmate/certs
sudo chmod 750 /srv/dexmate/certs

# 3. Place the current, validated certificate produced by the supported
#    Dexmate/dextop renewal flow
sudo cp /secure/path/to/VGE07DBE2D05.dzcfg /srv/dexmate/certs/
sudo chown root:dexlab /srv/dexmate/certs/VGE07DBE2D05.dzcfg
sudo chmod 640 /srv/dexmate/certs/VGE07DBE2D05.dzcfg

# 4. Add each lab member to the dexlab group
sudo usermod -aG dexlab <username>
```

Members must log out and back in after being added to `dexlab`.

The current `omniteleop` auto-detection only looks in
`~/.dexmate/comm/zenoh/`. When using the shared certificate, each user must
explicitly set:

```text
ZENOH_CONFIG=/srv/dexmate/certs/VGE07DBE2D05.dzcfg
```

---

# B. User-Level Configuration

Each lab member should run this section once in their own account.

## B.1 Prepare the Conda Environment

If `conda` is not already available in your account, source the shared
Miniforge initialization script first:

```bash
source /home/zinan/miniforge3/etc/profile.d/conda.sh
```

Create and activate the `dexrobot` environment:

```bash
conda create -n dexrobot python=3.11 -y
conda activate dexrobot
```

Install the required versions:

```bash
python -m pip install 'dextop==0.5.0'
python -m pip install 'dexcontrol==0.5.0'
```

Verify:

```bash
python --version
dextop --version
python -c "import importlib.metadata as m; print(m.version('dexcontrol'))"
```

Expected:

```text
Python 3.11
dextop 0.5.0
dexcontrol 0.5.0
```

If `dexrobot` already exists, check and upgrade `dexcontrol` if needed:

```bash
conda activate dexrobot
python -c "import importlib.metadata as m; print(m.version('dexcontrol'))"
python -m pip install 'dexcontrol==0.5.0'
```

## B.2 Set User Environment Variables

```bash
conda activate dexrobot

conda env config vars set \
  ROBOT_NAME='dm/vge07dbe2d05-1u' \
  ROBOT_IP='192.168.50.20' \
  ROBOT_CONFIG='vega_1u_gripper' \
  ZENOH_CONFIG='/srv/dexmate/certs/VGE07DBE2D05.dzcfg'

conda deactivate
conda activate dexrobot
```

Verify:

```bash
echo "$ROBOT_NAME"
echo "$ROBOT_IP"
echo "$ROBOT_CONFIG"
echo "$ZENOH_CONFIG"
```

Expected:

```text
dm/vge07dbe2d05-1u
192.168.50.20
vega_1u_gripper
/srv/dexmate/certs/VGE07DBE2D05.dzcfg
```

## B.3 Verify Workstation Communication

```bash
conda activate dexrobot
ping -c 2 192.168.50.20
dextop topic list --timeout 20
```

A healthy workstation should see topics such as:

```text
dm/vge07dbe2d05-1u/heartbeat
dm/vge07dbe2d05-1u/sensors/head_camera/imu
dm/vge07dbe2d05-1u/sensors/head_camera/left_rgb
dm/vge07dbe2d05-1u/sensors/head_camera/right_rgb
dm/vge07dbe2d05-1u/state/arm/left
dm/vge07dbe2d05-1u/state/arm/right
dm/vge07dbe2d05-1u/state/estop
dm/vge07dbe2d05-1u/state/gripper/left
dm/vge07dbe2d05-1u/state/gripper/right
dm/vge07dbe2d05-1u/state/head
dm/vge07dbe2d05-1u/state/wrench/left
dm/vge07dbe2d05-1u/state/wrench/right
dm/vge07dbe2d05-1u/state/wrist_button/left
dm/vge07dbe2d05-1u/state/wrist_button/right
```

You can also check the update rate of a read-only topic:

```bash
dextop topic hz dm/vge07dbe2d05-1u/state/arm/left
```

Press `Ctrl+C` to stop.

---

## Daily Startup Checklist

On the robot:

```bash
ssh dexmate@192.168.50.20
systemctl status dextop-node.service --no-pager
systemctl status dexsensor.service --no-pager
dexsensor status --robot dm/vge07dbe2d05-1u head_camera
```

On the workstation:

```bash
conda activate dexrobot
ping -c 2 192.168.50.20
dextop topic list --timeout 20
```

Do not run any motion example until the emergency stop, workspace clearance,
robot state, and the selected workflow have been reviewed.

---

## Troubleshooting

### Robot Does Not Reply to Ping

```bash
nmcli device status
ip -brief address show dev enp10s0
ip route get 192.168.50.20
```

Confirm the cable is connected to the torso port, the robot has finished
booting, and the workstation IP is `192.168.50.10/24`.

### SSH Works but Port 7447 Is Refused

The relay is not running. On the robot:

```bash
sudo systemctl restart dextop-node.service
systemctl status dextop-node.service --no-pager
```

### Relay Connects but Zero Topics Are Found

Check both sides use `dextop 0.5.0`:

```bash
dextop --version
```

If not, install the matching version:

```bash
python -m pip install 'dextop==0.5.0'
```

Then retry:

```bash
dextop topic list --timeout 20
```

### Only Head Camera Topics Are Found

If you only see:

```text
dm/vge07dbe2d05-1u/sensors/head_camera/imu
dm/vge07dbe2d05-1u/sensors/head_camera/left_rgb
dm/vge07dbe2d05-1u/sensors/head_camera/right_rgb
```

the relay is not routing robot state topics. On the robot:

```bash
systemctl status dextop-node.service --no-pager
journalctl -u dextop-node.service -n 80 --no-pager

env ROBOT_NAME=dm/vge07dbe2d05-1u \
  ZENOH_CONFIG=/home/dexmate/.dexmate/comm/zenoh/VGE07DBE2D05.dzcfg \
  /home/dexmate/miniconda3/envs/dexmate_env1/bin/dextop topic list --timeout 10
```

If the robot sees all topics but the workstation still sees only head camera
topics, restart the relay:

```bash
sudo systemctl restart dextop-node.service
```

### Head Camera Topics Are Missing

```bash
systemctl status dexsensor.service --no-pager
journalctl -u dexsensor.service -n 80 --no-pager
dexsensor status --robot dm/vge07dbe2d05-1u head_camera
```

Expected camera status:

```text
head_camera  zed_x_camera  Running  imu, left_rgb, right_rgb
```

### `dextop: command not found` on the Robot

```bash
conda activate dexmate_env1
```

### Certificate / TLS Errors

Confirm the shared certificate exists and has the correct permissions:

```bash
ls -l /srv/dexmate/certs/
```

Confirm your account is in the `dexlab` group:

```bash
groups
```

Confirm the environment variable:

```bash
echo "$ZENOH_CONFIG"
```

Also compare workstation and robot time. Zenoh TLS is sensitive to clock skew,
and a large time difference can cause connection failures.

### Multiple `dexcomm.Node` Instances

Each `Node(name=...)` creates an independent Zenoh session. All processes that
need to reach the same robot must use the same correct `ZENOH_CONFIG` and
`ROBOT_IP`. This is why the environment variables are stored in the
`dexrobot` conda environment instead of being exported manually each time.

---

## References

- [Dexmate Network Setup](https://docs.dexmate.ai/KS4XmA6JeiJ4zZfdUVw1/getting-started/network-setup)
- [Dexmate Software Setup and Updates](https://docs.dexmate.ai/KS4XmA6JeiJ4zZfdUVw1/getting-started/software-setup-and-updates)
- [Dexmate Documentation](https://docs.dexmate.ai/KS4XmA6JeiJ4zZfdUVw1)
