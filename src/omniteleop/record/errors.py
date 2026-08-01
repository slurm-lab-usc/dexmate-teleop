"""Recording errors that require an episode to be aborted."""


class RequiredSensorError(RuntimeError):
    """A required recording sensor is missing, stale, or unavailable."""
