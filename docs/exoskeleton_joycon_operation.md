# Exoskeleton + JoyCon Operation Guide

This guide describes how to operate the Dexmate Vega U with the exoskeleton
leader and two JoyCon controllers. It covers the startup flow, the web UI
states, and the exact button mappings.

## Scope

- Robot: Dexmate Vega U (`vega_1u_gripper` config)
- Leader: exoskeleton arms
- Controllers: left and right JoyCon
- Primary functions: arm teleoperation, gripper control, recording, E-stop

## Prerequisites

- The workstation and robot are configured according to the
  [Dexmate Vega U Workstation Setup](dexmate_vega_u_setup.md).
- Your `dexrobot` conda environment is ready and environment variables are set
  (see [Lab Member Quickstart](lab_member_quickstart.md)).
- The exoskeleton is connected to the workstation (e.g. via USB serial).
- Both JoyCons are paired and connected before launching the stack.

## Startup Workflow

### 1. Launch the backend

```bash
cd src/omniteleop/app
bash launch.sh
```

Open the web interface:

```text
http://localhost:5006
```

### 2. Start teleoperation

- Select **Leader Mode: Exoskeleton**.
- Optionally enable **Record Mode**.
- Click **Start**.

The backend starts the exoskeleton reader, JoyCon reader, command processor,
and robot controller.

### 3. Wait for the robot to initialize

The robot controller moves the arms to the configured home position first.
During this phase the UI may show `BOOT` or `DIAGNOSIS`.

### 4. Align the exoskeleton

When the UI shows **ALIGN**, follow the on-screen joint guide:

- Move the **exoskeleton arms**, not the JoyCons.
- Each joint row shows the exo position, robot position, target delta, and the
  direction to move.
- Keep the software E-Stop active until all 14 arm joints are ready.

Common alignment messages:

| Message | Meaning |
|---------|---------|
| `No exoskeleton joint data` | Check `arm_reader` and the exoskeleton USB/serial connection. |
| `No robot joint feedback` | Check `robot_controller` and the `robot/joints` topic. |
| `All joints are ready` | Keep the exoskeleton still and wait for `Paused`. |
| `Move the EXOSKELETON joints...` | Continue moving the physical exoskeleton in the indicated direction. |

### 5. Release E-stop

After alignment completes, the UI changes to **Paused** with E-Stop still
active.

To start motion:

- Press and hold **both JoyCon sticks** for about **1.5 seconds**, then release.

This toggles the software E-Stop. The UI should change to **Active** (or
**Record** if recording is enabled).

### 6. Operate

- Arms are controlled by the exoskeleton directly.
- Grippers and recording are controlled with the JoyCons (see button map
  below).

### 7. Stop

- Click **Stop** in the web interface, or
- Press `Ctrl+C` in the terminal running `launch.sh`.

The robot controller runs a safe shutdown path when stopping.

---

## JoyCon Button Reference

### Safety / System

| Action | Buttons | Duration |
|--------|---------|----------|
| Toggle software E-Stop | Both sticks pressed | ~1.5 s |
| Exit / shut down teleop | Left Capture + Right Home | ~1.0 s |

### Recording

| Action | Buttons | Notes |
|--------|---------|-------|
| Toggle recording | Hold Left L + Right R, then release both | The toggle commits on release. |
| Discard current recording | Hold Left ZL + Right ZR | Useful on robots without torso. |

### Grippers

| Action | Buttons |
|--------|---------|
| Left gripper fine open/close | Left stick up/down |
| Right gripper fine open/close | Right stick up/down |
| Left gripper full open | Left D-pad Up |
| Left gripper full close | Left D-pad Down |
| Right gripper full open | Right X |
| Right gripper full close | Right B |

Notes:

- `L` and `R` are **not** individual gripper actions.
- After an E-stop toggle, switching control modes, or using the `L + R`
  recording gesture, each gripper stick must return to center once before fine
  gripper control resumes.

### Other Toggles (Not Used on This Robot)

The following mappings exist in the controller but are not used for the
`vega_1u_gripper` upper-body setup:

| Action | Buttons |
|--------|---------|
| Toggle base control | Left Minus or Right Plus (single press) |
| Toggle head control | Left Minus + Right Plus together |
| Toggle torso control | Left ZL or Right ZR (single press) |

If your robot configuration has a head, use the `Minus + Plus` combo to enter
head control, then use the sticks to move the head.

---

## Detailed Workflow Example

1. Power on the workstation and robot.
2. Verify robot services:
   ```bash
   systemctl status dextop-node.service --no-pager
   systemctl status dexsensor.service --no-pager
   ```
3. Activate the environment:
   ```bash
   conda activate dexrobot
   ```
4. Start the backend:
   ```bash
   cd ~/workspace/projects/omniteleop/src/omniteleop/app
   bash launch.sh
   ```
5. Open `http://localhost:5006`.
6. Select **Exoskeleton**, enable recording if needed, and click **Start**.
7. Wait for homing to finish, then follow the ALIGN guide by moving the
   exoskeleton arms.
8. When the UI shows **Paused**, hold both sticks for 1.5 s to release E-Stop.
9. Use the exoskeleton for arm motion and the JoyCons for grippers/recording.
10. When finished, click **Stop** or press `Ctrl+C`.

---

## Safety Reminders

- Always keep an operator ready at the physical E-stop.
- Do not let two people operate the same robot at the same time.
- If the arms drift or a joint limit violation appears, stop immediately.
- A successful `disable` response does not prove that a mechanical brake is
  holding the arm. Watch for arm drift after shutdown or E-stop.
- If the exoskeleton data is missing, do not try to satisfy alignment with the
  JoyCons. Fix the exoskeleton connection first.
