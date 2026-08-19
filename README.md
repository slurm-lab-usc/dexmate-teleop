<div align="center">
  <h1>🎮 Omniteleop - Teleoperation Stack for Dexmate Robots</h1>
</div>

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

## 📦 Installation

```shell
cd omniteleop/
pip install -e .
```

## ✨ Features

- 🕹️ **JoyCon Controller Support** - Use Nintendo JoyCon for robot control
- 💪 **Exoskeleton Arm Control** - Intuitive arm teleoperation via Dynamixel exoskeleton
- 🥽 **VR Teleoperation** - Relative delta commands to conrol humanoid arm and torso.
- 🛡️ **Safety System** - Built-in emergency stop and joint limits enforcement
- 📹 **Data Collection** - Record teleoperation data for policy learning
- 🔄 **Trajectory Replay** - Replay recorded robot trajectories
- 📊 **Telemetry Viewer** - Real-time visualization of joint data
- 😎 **GUI** - Easy to use WebApp that provides an intuitive, visual layer to the entire repository.

## 🚀 Quick Start

```shell
omni-arm       # Exoskeleton arm reader
omni-joycon    # JoyCon controller reader
omni-cmd       # Command processor with safety
omni-robot     # Robot controller
omni-recorder  # MDP recorder for policy learning
omni-mcap-recorder # MCAP recorder for policy learning
omni-wrist-cameras # Local UVC wrist cameras → standard Zenoh camera topics
omni-telemetry # Telemetry viewer
omni-paddle    # VR reader

app/launch.sh  # One command that replaces all
```

### JoyCon Gripper Controls

For gripper robot configurations such as `vega_1u_gripper`:

- Left/right stick up and down: proportional fine open/close for that gripper.
- Left D-pad Up/Down: fully open/close the left gripper.
- Right X/B: fully open/close the right gripper.
- Hold L+R for `recording_hold_duration`, then release both: toggle recording.
- L and R have no individual gripper action.

After E-stop, another control mode, or the L+R recording gesture, each gripper
stick must return to center once before fine control resumes.

### Local UVC Wrist Cameras

When `recorder.wrist_camera_adapter.enabled` and either wrist RGB component are
enabled in the active robot YAML, the app starts the adapter automatically in
record mode. It publishes `/dev/dexmate-wrist-left` and
`/dev/dexmate-wrist-right` on the standard Dexmate wrist-camera topics. For a
standalone stream check, run:

```shell
omni-wrist-cameras
```

## 📚 Documentation

- [Dexmate Vega U Workstation Setup](docs/dexmate_vega_u_setup.md)
- [Lab Member Quickstart](docs/lab_member_quickstart.md)

## 📄 Licensing

This project is **dual-licensed**:

### 🔓 Open Source License
This software is available under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
See the [LICENSE](./LICENSE) file for details.

### 💼 Commercial License
For businesses that want to use this software in proprietary applications without the AGPL requirements, commercial licenses are available.

**📧 Contact us for commercial licensing:** contact@dexmate.ai

---

<div align="center">
  <h3>🤝 Ready to teleoperate robots?</h3>
  <p>
    <a href="mailto:contact@dexmate.ai">📧 Contact Us</a>
  </p>
</div>
