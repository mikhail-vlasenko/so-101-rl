"""Minimal native wrapper over scservo_sdk for the SO-101 servos.

Replaces vassar_feetech_servo_sdk.ServoController with only the methods
needed by the digital-twin tool. Fail-fast on any COMM error.
"""

import sys
from pathlib import Path

import numpy as np

SDK_DIR = Path(__file__).resolve().parent.parent.parent.parent / "feetech-servo-sdk"
sys.path.insert(0, str(SDK_DIR))

import scservo_sdk as scs  # noqa: E402

ADDR_TORQUE_ENABLE = 40
ADDR_POSITION_KP = 21
ADDR_CW_DEADZONE = 26
ADDR_CCW_DEADZONE = 27
ADDR_TORQUE_LIMIT = 48
ADDR_PRESENT_POSITION = 56
LEN_PRESENT_POSITION = 2
BAUDRATE = 1_000_000


class ServoBus:
    def __init__(self, port: str, servo_ids: list[int], allow_missing: bool = False):
        self.port = port
        self.servo_ids = list(servo_ids)
        self.allow_missing = allow_missing
        self.present_mask: np.ndarray = np.ones(len(self.servo_ids), dtype=bool)
        self.port_handler: scs.PortHandler | None = None
        self.packet_handler: scs.sms_sts | None = None
        self.sync_read: scs.GroupSyncRead | None = None
        # Last-known raw read per servo; missing slots stay at 2048 so the
        # viewer renders them at their mechanical midpoint.
        self._last_read: np.ndarray = np.full(len(self.servo_ids), 2048, dtype=np.int64)

    def connect(self) -> None:
        ph = scs.PortHandler(self.port)
        if not ph.openPort():
            raise RuntimeError(f"failed to open port {self.port}")
        if not ph.setBaudRate(BAUDRATE):
            ph.closePort()
            raise RuntimeError(f"failed to set baudrate {BAUDRATE} on {self.port}")
        self.port_handler = ph
        self.packet_handler = scs.sms_sts(ph)

        # Ping each servo to discover which are physically present. Without
        # --allow-missing this is just an early sanity check (one clear error
        # per missing ID instead of a cryptic sync-read failure later).
        present = np.zeros(len(self.servo_ids), dtype=bool)
        for i, sid in enumerate(self.servo_ids):
            _, result, _ = self.packet_handler.ping(sid)
            present[i] = (result == scs.COMM_SUCCESS)
        missing = [sid for sid, ok in zip(self.servo_ids, present) if not ok]
        if missing and not self.allow_missing:
            ph.closePort()
            self.port_handler = None
            self.packet_handler = None
            raise RuntimeError(
                f"servos not responding to ping: {missing}. "
                f"Re-run with --allow-missing to continue without them."
            )
        self.present_mask = present
        if missing:
            print(f"[ServoBus] allow_missing: skipping servos {missing}; "
                  f"active set = {[s for s, ok in zip(self.servo_ids, present) if ok]}")

        self.sync_read = scs.GroupSyncRead(
            self.packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        for sid, ok in zip(self.servo_ids, self.present_mask):
            if not ok:
                continue
            if not self.sync_read.addParam(sid):
                raise RuntimeError(f"failed to add servo {sid} to sync read")

    def close(self) -> None:
        if self.port_handler is None:
            return
        self.disable_torque_all()
        self.port_handler.closePort()
        self.port_handler = None
        self.packet_handler = None
        self.sync_read = None

    def _set_torque(self, value: int) -> None:
        assert self.packet_handler is not None, "ServoBus not connected"
        for sid, ok in zip(self.servo_ids, self.present_mask):
            if not ok:
                continue
            result, error = self.packet_handler.write1ByteTxRx(sid, ADDR_TORQUE_ENABLE, value)
            if result != scs.COMM_SUCCESS:
                raise RuntimeError(
                    f"servo {sid} torque write failed: "
                    f"{self.packet_handler.getTxRxResult(result)}"
                )
            if error != 0:
                raise RuntimeError(
                    f"servo {sid} torque write error: "
                    f"{self.packet_handler.getRxPacketError(error)}"
                )

    def enable_torque_all(self) -> None:
        self._set_torque(1)

    def disable_torque_all(self) -> None:
        self._set_torque(0)

    def set_position_kp(self, kp_per_servo) -> None:
        """Write position-loop proportional gain (addr 21) per servo (in
        servo_ids order). Factory default is 32; raising it makes the servo
        apply more torque per unit of position error, helping overcome static
        friction at small commanded deltas. Range 0..254 per value.
        Gravity-loaded joints (shoulder_lift, elbow_flex) need higher Kp; wrist
        joints oscillate at high Kp, so tune per joint."""
        assert self.packet_handler is not None, "ServoBus not connected"
        kp_list = list(kp_per_servo)
        assert len(kp_list) == len(self.servo_ids), (
            f"expected {len(self.servo_ids)} Kp values, got {len(kp_list)}"
        )
        for sid, kp, ok in zip(self.servo_ids, kp_list, self.present_mask):
            if not ok:
                continue
            assert 0 <= kp <= 254, f"servo {sid} kp out of range: {kp}"
            result, error = self.packet_handler.write1ByteTxRx(sid, ADDR_POSITION_KP, int(kp))
            if result != scs.COMM_SUCCESS:
                raise RuntimeError(
                    f"servo {sid} Kp write failed: "
                    f"{self.packet_handler.getTxRxResult(result)}"
                )
            if error != 0:
                raise RuntimeError(
                    f"servo {sid} Kp write error: "
                    f"{self.packet_handler.getRxPacketError(error)}"
                )

    def set_position_deadzone(self, deadzone_per_servo) -> None:
        """Write the symmetric position-correction deadzone (registers 26 / 27,
        CW / CCW) per servo. Units: 0.087°/unit, range 0..16, factory default 1.
        The servo's position loop will NOT try to correct when |error| is within
        ±deadzone, letting the joint settle anywhere inside a backlash gap
        without the PID hunting across it. Mitigates load-dependent chatter on
        gravity- or moment-loaded joints (shoulder_pan in particular).
        Registers are EEPROM and persist across power cycles, so this value
        outlives the running program until something else writes them."""
        assert self.packet_handler is not None, "ServoBus not connected"
        dz_list = list(deadzone_per_servo)
        assert len(dz_list) == len(self.servo_ids), (
            f"expected {len(self.servo_ids)} deadzone values, got {len(dz_list)}"
        )
        for sid, dz, ok in zip(self.servo_ids, dz_list, self.present_mask):
            if not ok:
                continue
            assert 0 <= dz <= 16, f"servo {sid} deadzone out of range: {dz}"
            for addr in (ADDR_CW_DEADZONE, ADDR_CCW_DEADZONE):
                result, error = self.packet_handler.write1ByteTxRx(sid, addr, int(dz))
                if result != scs.COMM_SUCCESS:
                    raise RuntimeError(
                        f"servo {sid} deadzone write (addr {addr}) failed: "
                        f"{self.packet_handler.getTxRxResult(result)}"
                    )
                if error != 0:
                    raise RuntimeError(
                        f"servo {sid} deadzone write (addr {addr}) error: "
                        f"{self.packet_handler.getRxPacketError(error)}"
                    )

    def set_torque_limit(self, limit: int) -> None:
        """Write the output-torque cap (RAM register 48, 2 bytes) to every
        connected servo: 0-1000 = fraction of stall torque. RAM-only — the
        firmware re-initializes it from EPROM Max_Torque_Limit (addr 16) at
        power-on, so every torque-on session must re-write it (boot paths do).
        Keep it in sync with the sim actuator forcerange; see
        real/twin/constants.SERVO_TORQUE_LIMIT_FRAC."""
        assert self.packet_handler is not None, "ServoBus not connected"
        assert 0 < limit <= 1000, f"torque limit out of range: {limit}"
        for sid, ok in zip(self.servo_ids, self.present_mask):
            if not ok:
                continue
            result, error = self.packet_handler.write2ByteTxRx(
                sid, ADDR_TORQUE_LIMIT, int(limit))
            if result != scs.COMM_SUCCESS:
                raise RuntimeError(
                    f"servo {sid} torque-limit write failed: "
                    f"{self.packet_handler.getTxRxResult(result)}"
                )
            if error != 0:
                raise RuntimeError(
                    f"servo {sid} torque-limit write error: "
                    f"{self.packet_handler.getRxPacketError(error)}"
                )

    def read_pos_speed(self, servo_id: int) -> tuple[int, int]:
        """Read present position + present speed of ONE servo in a single bus
        transaction, and return the speed too — which the position-only read_all
        sync read does not. A single-servo round trip is a fraction of the full
        6-servo sync read (the latter measures ~1.4 ms at 1 Mbaud in the rollout
        loop), fast enough to sample one joint at several hundred Hz — enough to
        resolve mechanical resonances that alias at the 15 Hz control rate. Speed
        is the servo's reported signed raw value."""
        assert self.packet_handler is not None, "ServoBus not connected"
        pos, speed, result, error = self.packet_handler.ReadPosSpeed(servo_id)
        if result != scs.COMM_SUCCESS:
            raise RuntimeError(
                f"servo {servo_id} pos/speed read failed: "
                f"{self.packet_handler.getTxRxResult(result)}"
            )
        if error != 0:
            raise RuntimeError(
                f"servo {servo_id} pos/speed read error: "
                f"{self.packet_handler.getRxPacketError(error)}"
            )
        return pos, speed

    def read_all(self) -> np.ndarray:
        """Read present position for every connected servo. Returns int64 array
        in servo_ids order; entries for missing servos hold the placeholder
        2048 (mechanical midpoint) so downstream consumers don't need to know."""
        assert self.sync_read is not None and self.packet_handler is not None, "ServoBus not connected"
        if not self.present_mask.any():
            return self._last_read.copy()
        result = self.sync_read.txRxPacket()
        if result != scs.COMM_SUCCESS:
            raise RuntimeError(
                f"sync read failed: {self.packet_handler.getTxRxResult(result)}"
            )
        for i, (sid, ok) in enumerate(zip(self.servo_ids, self.present_mask)):
            if not ok:
                continue
            available, _ = self.sync_read.isAvailable(
                sid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            )
            if not available:
                raise RuntimeError(f"servo {sid} not available in sync read response")
            raw = self.sync_read.getData(sid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            self._last_read[i] = self.packet_handler.scs_tohost(raw, 15)
        return self._last_read.copy()

    def write_all(self, raw: np.ndarray, speed: int, accel: int) -> None:
        """Sync-write goal positions to every connected servo (in servo_ids
        order). Entries for missing servos are skipped."""
        assert self.packet_handler is not None, "ServoBus not connected"
        assert raw.shape == (len(self.servo_ids),), f"raw shape {raw.shape}"
        if not self.present_mask.any():
            return
        gsw = self.packet_handler.groupSyncWrite
        gsw.clearParam()
        for sid, pos, ok in zip(self.servo_ids, raw.tolist(), self.present_mask):
            if not ok:
                continue
            if not self.packet_handler.SyncWritePosEx(sid, int(pos), speed, accel):
                gsw.clearParam()
                raise RuntimeError(f"servo {sid} addParam failed in sync write")
        result = gsw.txPacket()
        gsw.clearParam()
        if result != scs.COMM_SUCCESS:
            raise RuntimeError(
                f"sync write failed: {self.packet_handler.getTxRxResult(result)}"
            )
