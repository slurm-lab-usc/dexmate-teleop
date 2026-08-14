#!/usr/bin/env bash
# Launch the OmniTeleop backend.  The frontend is served by the backend itself
# at http://localhost:5006 — no separate npm/node process is needed.
#
# Usage:
#   ./launch.sh                                  # normal mode
#   ./launch.sh --rpi-mode                       # RPi leader mode (bypasses local hw checks)
#   ./launch.sh --lock-hand-collision-avoidance  # freeze hands in MotionManager
#                                                # collision model at open pose
#   ./launch.sh --unlock-torso                   # let paddle_leader IK recruit torso
#                                                # joints (vr / vr_sim modes; off by
#                                                # default so the same launch works
#                                                # on upper-body-only vega_1u)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

RPI_FLAG=""
LOCK_HAND_FLAG=""
UNLOCK_TORSO_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rpi-mode) RPI_FLAG="--rpi-mode"; shift ;;
    --lock-hand-collision-avoidance) LOCK_HAND_FLAG="--lock-hand-collision-avoidance"; shift ;;
    --unlock-torso) UNLOCK_TORSO_FLAG="--unlock-torso"; shift ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--rpi-mode] [--lock-hand-collision-avoidance] [--unlock-torso]"
      exit 1
      ;;
  esac
done

BACKEND_PID=""

cleanup() {
  echo ""
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "→ Stopping backend (pid $BACKEND_PID)..."
    kill -TERM "$BACKEND_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$BACKEND_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "$BACKEND_PID" 2>/dev/null || true
  fi
  echo "→ Done."
}
trap cleanup EXIT INT TERM

# ── Kill stale teleop processes from previous runs ───────────────────────────
# Covers every subprocess spawned by app_backend._build_teleop_commands across
# all three leader modes:
#   exoskeleton → arm_reader, joycon_reader, command_processor, robot_controller
#   vr          → paddle_leader, robot_controller
#   vr_sim      → paddle_leader
# Plus the recorder subprocesses (mcap_recorder, mdp_recorder), the replay
# publisher (replay_record), and the wrist camera adapter, which run as
# `python -m omniteleop.record.<module>` so we match the module path too.
STALE_PATTERN="joycon_reader\.py|arm_reader\.py|robot_controller\.py|command_processor\.py|paddle_leader\.py|omniteleop\.record\.((mcap|mdp)_recorder|replay_record|wrist_camera_adapter)"

STALE_PIDS=$(pgrep -f "$STALE_PATTERN" || true)
if [[ -n "$STALE_PIDS" ]]; then
  echo "→ Killing stale teleop processes: $STALE_PIDS"
  # shellcheck disable=SC2086
  kill -TERM $STALE_PIDS 2>/dev/null || true
  for _ in $(seq 1 10); do
    pgrep -f "$STALE_PATTERN" >/dev/null || break
    sleep 0.5
  done
  REMAINING=$(pgrep -f "$STALE_PATTERN" || true)
  if [[ -n "$REMAINING" ]]; then
    echo "→ Force-killing leftovers: $REMAINING"
    # shellcheck disable=SC2086
    kill -KILL $REMAINING 2>/dev/null || true
  fi
fi

# ── Backend ───────────────────────────────────────────────────────────────────
echo "→ Starting backend${RPI_FLAG:+ (rpi-mode)}..."
cd "$REPO_ROOT"
export ROBOT_CONFIG="${ROBOT_CONFIG:-vega_1u_gripper}"
python -m omniteleop.app.backend.app_backend $RPI_FLAG $LOCK_HAND_FLAG $UNLOCK_TORSO_FLAG &
BACKEND_PID=$!

echo ""
echo "  Backend  PID : $BACKEND_PID"
echo "  App URL      : http://localhost:5006"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

wait "$BACKEND_PID"
