"""Bounded real servo reads, crediting the joint ID that actually replied."""
import math
import time


LIMITS = (180, 180, 180, 180, 270, 180)


def raw_to_degrees(servo_id, raw):
    # Same conversion as Rosmaster and v23's _rosmaster_raw_to_deg.
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if servo_id in (1, 2, 3, 4):
        degrees = int((value - 900) * -180 / 2200 + 180 + .5)
    elif servo_id == 5:
        degrees = int(270 * (value - 380) / 3320 + .5)
    elif servo_id == 6:
        degrees = int(180 * (value - 900) / 2200 + .5)
    else:
        return None
    return float(degrees) if 0 <= degrees <= LIMITS[servo_id-1] else None


def valid_angle(value, index):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= LIMITS[index])


def read_angles(device, attempts=4):
    """Caller must hold the shared serial lock. No simulated/last-command fallback."""
    angles = [None] * 6
    array_getter = getattr(device, "get_uart_servo_angle_array", None)
    raw_getter = getattr(device, "get_uart_servo_value", None)
    replies = []
    for attempt in range(attempts):
        if attempt:
            time.sleep(.06)
        missing = [i for i in range(6) if angles[i] is None]
        if not missing:
            return angles
        if array_getter and len(missing) > 1:
            try:
                values = array_getter()
                if values is not None and len(values) == 6:
                    for i in missing:
                        if valid_angle(values[i], i):
                            angles[i] = float(values[i])
            except Exception:
                pass
        for i in range(6):
            if angles[i] is not None:
                continue
            try:
                if raw_getter:
                    answer, raw = raw_getter(i+1)
                    replies.append((i+1, answer, raw))
                    if not isinstance(answer, int) or not 1 <= answer <= 6:
                        continue
                    value = raw_to_degrees(answer, raw)
                    index = answer-1
                else:
                    value = device.get_uart_servo_angle(i+1)
                    index = i
                if valid_angle(value, index):
                    angles[index] = float(value)
            except Exception:
                pass
        if all(a is not None for a in angles):
            return angles
    missing = ["S" + str(i+1) for i, a in enumerate(angles) if a is None]
    raise ValueError("servo_read_failed: " + ",".join(missing) + "; raw_replies=" + repr(replies))
