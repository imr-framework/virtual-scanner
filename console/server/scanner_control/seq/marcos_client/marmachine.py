#/usr/bin/env python3
# Machine code defines and functions for marga.
#
# Functions should be fast, without any floating-point arithmetic -
# that should be handled at a higher level.

class MarUserWarning(UserWarning):
    pass

class MarCompileWarning(MarUserWarning):
    pass

class MarRemovedInstructionWarning(MarCompileWarning):
    pass

class MarGradWarning(MarUserWarning):
    pass

class MarServerWarning(MarUserWarning):
    pass

INOP = 0x0
IFINISH = 0x1
IWAIT = 0x2
ITRIG = 0x3
ITRIGFOREVER=0x4
IDATA = 0x80

GRAD_CTRL = 0
GRAD_LSB = 1
GRAD_MSB = 2
RX0_RATE = 3
RX1_RATE = 4
TX0_I = 5
TX0_Q = 6
TX1_I = 7
TX1_Q = 8
DDS0_PHASE_LSB = 9
DDS0_PHASE_MSB = 10
DDS1_PHASE_LSB = 11
DDS1_PHASE_MSB = 12
DDS2_PHASE_LSB = 13
DDS2_PHASE_MSB = 14
GATES_LEDS = 15
RX_CTRL = 16
MARGA_BUFS = RX_CTRL + 1

STATE_IDLE = 0
STATE_PREPARE = 1
STATE_RUN = 2
STATE_COUNTDOWN = 3
STATE_TRIG = 4
STATE_TRIG_FOREVER = 5
STATE_HALT = 8

COUNTER_MAX = 0xffffff

CIC_STAGES = 6 # N: number of CIC stages in the RX CICs
# diff_delay = 1 # M: differential delay in comb section of CICs
CIC_RATE_DATAWIDTH = 12 # 12-bit rate/data bus, 2-bit address
CIC_FASTEST_RATE, CIC_SLOWEST_RATE = 4, 4095 # CIC core settings

def insta(instr, data):
    """ Instruction A: FSM control """
    assert instr in [INOP, IFINISH, IWAIT, ITRIG, ITRIGFOREVER], "Unknown instruction"
    assert (data & COUNTER_MAX) == (data & 0xffffffff), "Data out of range"
    return (instr << 24) | (data & 0xffffff)

def instb(tgt, delay, data):
    """ Instruction B: timed buffered data """
    assert tgt <= 24, "Unknown target buffer"
    assert 0 <= delay <= 255, "Delay out of range"
    assert (data & 0xffff) == (data & 0xffffffff), "Data out of range"
    return (IDATA << 24) | ( (tgt & 0x7f) << 24 ) | ( (delay & 0xff) << 16 ) | (data & 0xffff)
