# Omniteleop

Omniteleop is a teleoperation stack for Dexmate robots. It provides the
software needed to control a Dexmate Vega U robot with an exoskeleton leader,
collect teleoperation data, replay trajectories, and monitor robot state
through a web interface.

The primary workflow in this lab is **exoskeleton teleoperation**.

## Documentation

Before using the robot, follow these guides to configure the workstation and
your user environment:

- [Dexmate Vega U Workstation Setup](https://github.com/Buzzy0423/omniteleop/blob/main/docs/dexmate_vega_u_setup.md)
- [Lab Member Quickstart](https://github.com/Buzzy0423/omniteleop/blob/main/docs/lab_member_quickstart.md)

The setup guide covers network, robot-side services, the shared certificate,
and user-level environment variables. The quickstart covers cloning the repo,
creating the `dexrobot` conda environment, setting `ROBOT_NAME` / `ROBOT_IP` /
`ROBOT_CONFIG` / `ZENOH_CONFIG`, and verifying robot connectivity.

## Features

- Exoskeleton arm teleoperation
- JoyCon controller support
- Safety system with emergency stop and joint-limit enforcement
- Data collection / recording for policy learning
- Trajectory replay
- Web-based GUI for teleop, monitoring, and recording
- Local UVC wrist camera support

## Installation

```bash
cd omniteleop/
pip install -e .
```

For the full multi-user setup, including the correct `dexcontrol` and `dextop`
versions, see the [Lab Member Quickstart](https://github.com/Buzzy0423/omniteleop/blob/main/docs/lab_member_quickstart.md).

## Teleop Workflow

### 1. Prepare the environment

Make sure you have completed the steps in the
[Lab Member Quickstart](https://github.com/Buzzy0423/omniteleop/blob/main/docs/lab_member_quickstart.md):

- `dexrobot` conda environment created
- `dextop==0.4.7` and `dexcontrol==0.4.10` installed
- `ROBOT_NAME`, `ROBOT_IP`, `ROBOT_CONFIG`, and `ZENOH_CONFIG` set
- Robot connection verified with `dextop topic list`

### 2. Start the teleop backend

```bash
cd src/omniteleop/app
bash launch.sh
```

### 3. Open the web interface

Navigate to:

```text
http://localhost:5006
```

### 4. Start exoskeleton teleoperation

- In the web interface, select **Leader Mode: Exoskeleton**.
- Follow the alignment steps shown in the UI.
- Once the robot is aligned and active, use the exoskeleton to control the
  robot.

### 5. Stop teleoperation

- Click **Stop** in the web interface, or
- Press `Ctrl+C` in the terminal running `launch.sh`.

The system is designed to run a safe shutdown path when stopping.

## Recording Data

Enable **Record Mode** in the web interface before starting teleoperation, or
use the JoyCon recording gesture. Recorded episodes are saved according to the
active robot YAML configuration (`recorder.save_dir`).

## Command-Line Tools

The repository also provides individual command-line entry points:

```text
omni-arm            Exoskeleton arm reader
omni-joycon         JoyCon controller reader
omni-cmd            Command processor with safety
omni-robot          Robot controller
omni-recorder       MDP recorder for policy learning
omni-mcap-recorder  MCAP recorder for policy learning
omni-wrist-cameras  Local UVC wrist cameras
omni-telemetry      Telemetry viewer
```

## License

This project is dual-licensed:

- **Open Source**: GNU Affero General Public License v3.0 (AGPL-3.0)
- **Commercial**: available from Dexmate

See [LICENSE](./LICENSE) for details.
