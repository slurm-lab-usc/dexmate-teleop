# Dexmate Vega U Lab Member Quickstart

This guide helps lab members start using the shared Dexmate Vega U workstation
with exoskeleton teleoperation.

## Prerequisites

- You have a Linux account on the shared workstation.
- Your account is a member of the `dexlab` group so you can read the shared
  certificate.
- The robot-side services are already running:
  - `dextop-node.service`
  - `dexsensor.service`
- The workstation network is already configured for direct connection to the
  robot at `192.168.50.20`.

## 1. Get the Code

Create a projects directory and clone the repositories you need.

```bash
mkdir -p ~/workspace/projects
cd ~/workspace/projects

# Teleoperation / data collection
git clone git@github.com:slurm-lab-usc/dexmate-teleop.git

# Policy rollout
git clone git@github.com:Buzzy0423/robot-policy-deploy.git
```

## 2. Set Up the Conda Environment

If `conda` is not already available in your account, source the shared
Miniforge initialization script:

```bash
source /home/zinan/miniforge3/etc/profile.d/conda.sh
```

Create and activate `dexrobot`:

```bash
conda create -n dexrobot python=3.11 -y
conda activate dexrobot
```

Install `omniteleop`:

```bash
cd ~/workspace/projects/dexmate-teleop
python -m pip install -e .
```

Install the required communication versions:

```bash
python -m pip install 'dextop==0.5.0'
python -m pip install 'dexcontrol==0.5.0'
```

Verify:

```bash
python -c "import importlib.metadata as m; print(m.version('dexcontrol'))"
# Expected: 0.4.10
```

## 3. Set User Environment Variables

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

## 4. Verify the Robot Connection

```bash
conda activate dexrobot
ping -c 2 192.168.50.20
dextop topic list --timeout 20
```

You should see robot state topics under `dm/vge07dbe2d05-1u/...`.

The robot certificate expires approximately every 90 days. If the robot still
replies to `ping` but `dextop topic list` fails with a certificate/TLS error or
shows no robot topics, contact an administrator to check the expiration date.
Administrators renew the certificate through the internal management UI at
`http://192.168.50.21:57832`; lab members should not attempt to replace or
unpack certificate files themselves.

## 5. Start Exoskeleton Teleoperation

```bash
cd ~/workspace/projects/dexmate-teleop/src/omniteleop/app
bash launch.sh
```

After startup:

- Open the web interface at http://localhost:5006
- Select **Leader Mode: Exoskeleton**
- Follow the on-screen alignment steps, then start teleoperation

To stop:

- Click **Stop** in the web interface, or press `Ctrl+C` in the terminal
  running `launch.sh`.

## 6. Data Collection (Optional)

Enable **Record Mode** in the web interface when starting teleoperation, or use
the JoyCon recording gesture.

Recorded episodes can include:

- Head camera left/right RGB
- Wrist camera left/right RGB
- Robot joint states (arms, grippers, head)
- Left/right wrist force/torque
- Commanded actions

Data is saved under the `recorder.save_dir` setting in the active
`omniteleop` YAML configuration. The default format is MCAP; MDP is also
available. See the repository README for the full storage layout.

## 7. Policy Rollout (Optional)

If you need to run policy rollout:

```bash
cd ~/workspace/projects/robot-policy-deploy
conda run --no-capture-output -n dexrobot ./tools/deploy inspect --config configs/dexmate_qpose.yaml
conda run --no-capture-output -n dexrobot ./tools/deploy session --config configs/dexmate_qpose.yaml
```

More details are available in the `robot-policy-deploy` repository README and
docs.

## 8. Safety Notes

- Confirm the area around the robot is clear and the emergency stop is
  functional before starting.
- Do not let two people connect to the robot at the same time.
- If an arm reports a joint limit violation or any abnormal state, stop
  immediately and contact the administrator.
- A successful `disable` response does not prove that a mechanical brake is
  holding the arm. After shutdown or E-stop, still watch for arm drift.

## 9. Further Reading

- [Dexmate Vega U Workstation Setup](https://github.com/slurm-lab-usc/dexmate-teleop/blob/main/docs/dexmate_vega_u_setup.md)
- [Exoskeleton + JoyCon Operation Guide](https://github.com/slurm-lab-usc/dexmate-teleop/blob/main/docs/exoskeleton_joycon_operation.md)
- [robot-policy-deploy](https://github.com/Buzzy0423/robot-policy-deploy)
