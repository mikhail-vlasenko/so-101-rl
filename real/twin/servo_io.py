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
ADDR_PRESENT_POSITION = 56
LEN_PRESENT_POSITION = 2
BAUDRATE = 1_000_000


class ServoBus:
    def __init__(self, port: str, servo_ids: list[int]):
        self.port = port
        self.servo_ids = list(servo_ids)
        self.port_handler: scs.PortHandler | None = None
        self.packet_handler: scs.sms_sts | None = None
        self.sync_read: scs.GroupSyncRead | None = None

    def connect(self) -> None:
        ph = scs.PortHandler(self.port)
        if not ph.openPort():
            raise RuntimeError(f"failed to open port {self.port}")
        if not ph.setBaudRate(BAUDRATE):
            ph.closePort()
            raise RuntimeError(f"failed to set baudrate {BAUDRATE} on {self.port}")
        self.port_handler = ph
        self.packet_handler = scs.sms_sts(ph)
        self.sync_read = scs.GroupSyncRead(
            self.packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        for sid in self.servo_ids:
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
        for sid in self.servo_ids:
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

    def read_all(self) -> np.ndarray:
        """Read present position for every configured servo. Returns int64 array in servo_ids order."""
        assert self.sync_read is not None and self.packet_handler is not None, "ServoBus not connected"
        result = self.sync_read.txRxPacket()
        if result != scs.COMM_SUCCESS:
            raise RuntimeError(
                f"sync read failed: {self.packet_handler.getTxRxResult(result)}"
            )
        out = np.zeros(len(self.servo_ids), dtype=np.int64)
        for i, sid in enumerate(self.servo_ids):
            available, _ = self.sync_read.isAvailable(
                sid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            )
            if not available:
                raise RuntimeError(f"servo {sid} not available in sync read response")
            raw = self.sync_read.getData(sid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            out[i] = self.packet_handler.scs_tohost(raw, 15)
        return out

    def write_all(self, raw: np.ndarray, speed: int, accel: int) -> None:
        """Sync-write goal positions to every servo (in servo_ids order)."""
        assert self.packet_handler is not None, "ServoBus not connected"
        assert raw.shape == (len(self.servo_ids),), f"raw shape {raw.shape}"
        gsw = self.packet_handler.groupSyncWrite
        gsw.clearParam()
        for sid, pos in zip(self.servo_ids, raw.tolist()):
            if not self.packet_handler.SyncWritePosEx(sid, int(pos), speed, accel):
                gsw.clearParam()
                raise RuntimeError(f"servo {sid} addParam failed in sync write")
        result = gsw.txPacket()
        gsw.clearParam()
        if result != scs.COMM_SUCCESS:
            raise RuntimeError(
                f"sync write failed: {self.packet_handler.getTxRxResult(result)}"
            )
