#!/usr/bin/python3
# Copyright (C) 2013-2014, Jonathan A. Booth
#
# Copyrights belong to me, based on Ticket 6711044 which clarified this
# work is not assigned to my employer -- January 27, 2014.
"""An API to interface with USB PlasmaTrim devices.

>>> import plasmatrim
>>> plasmas = plasmatrim.find()
>>> len(plasmas)
1
>>> pt = plasmas[0]
>>> print(pt) # doctest: +ELLIPSIS
<PlasmaTrim RGB-8 ...>
>>> print(repr(pt))
PlasmaTrim('/dev/hidraw1')
>>> print(pt.name) # doctest: +ELLIPSIS
PlasmaTrim RGB-8 ...
>>> print('Serial %s' % hex(pt.serial)) # doctest: +ELLIPSIS
Serial ...
>>> print(pt.program)
Program:
  <Leds F00000, F04000, F0F000, 00F000, 009090, 0000F0, 8000F0, C000C0 1/0>
  <Leds C000C0, F00000, F04000, F0F000, 00F000, 009090, 0000F0, 8000F0 1/0>
  <Leds 8000F0, C000C0, F00000, F04000, F0F000, 00F000, 009090, 0000F0 1/0>
  <Leds 0000F0, 8000F0, C000C0, F00000, F04000, F0F000, 00F000, 009090 1/0>
  <Leds 009090, 0000F0, 8000F0, C000C0, F00000, F04000, F0F000, 00F000 1/0>
  <Leds 00F000, 009090, 0000F0, 8000F0, C000C0, F00000, F04000, F0F000 1/0>
  <Leds F0F000, 00F000, 009090, 0000F0, 8000F0, C000C0, F00000, F04000 1/0>
  <Leds F00000, F0F000, 00F000, 009090, 0000F0, 8000F0, C000C0, F00000 1/0>
>>> pt.start()
>>> import time
>>> time.sleep(2)
>>> pt.stop()
>>> pt.program.run(2)
>>> for i in range(len(pt.leds)):
...   prg = pt.program[i]
...   for j in range(len(pt.leds)):
...     prg.leds[j].red = 0
...     prg.leds[j].green = 0
...     prg.leds[j].blue = 0
...   prg.leds[i].blue = 255
...   prg.hold = 2
>>> pt.program[len(pt.program):] = pt.program[-2:0:-1]
>>> pt.program.run(2)
>>> for i in range(len(pt.leds)):
...   pt.leds[i].red = 255
...   pt.leds[i].green = 0
...   pt.leds[i].blue = 0
>>> pt.leds.show()
>>> print(pt.leds)
<Leds FF0000, FF0000, FF0000, FF0000, FF0000, FF0000, FF0000, FF0000>
>>> for i in range(100, 0, -1):
...   pt.brightness = i
...   pt.leds.show()
>>> pt.brightness
1
"""

__all__ = ['find', 'PlasmaTrim']

from binascii import hexlify
from colorsys import hsv_to_rgb, rgb_to_hsv
from time import sleep, time
import colors
import glob
import logging
import re
import signal
import struct
import sys

# Compiled regex to identify a plasmatrim in /sys/class/hidraw
HID_ID = re.compile('^HID_ID=0003:000026F3:00001000$')

# Logger
LOG = logging.getLogger(__name__)

def find():
    """Returns a list of PlasmaTrim objects, each connected to a
    different real PlasmaTrim device."""
    paths = glob.glob('/sys/class/hidraw/*/device/uevent')
    devices = []
    for path in paths:
        with open(path, 'r') as dev:
            for line in dev:
                if HID_ID.match(line):
                    chunks = path.split('/')
                    devices.append(chunks[4])
    return [PlasmaTrim("/dev/%s" % dev) for dev in devices]

class _TimeoutError(IOError):
    """Used to cause a timeout in the send function."""
    @staticmethod
    def timeout(signum, frame):
        """Called when a timeout happens."""
        raise _TimeoutError()

# PlasmaTrim command names and way to access them
COMMANDS = [ 'IMMEDIATE_WRITE', 'IMMEDIATE_READ',
             'START_SEQUENCE', 'STOP_SEQUENCE',
             'WRITE_TABLE_LENGTH', 'READ_TABLE_LENGTH',
             'WRITE_TABLE_ENTRY', 'READ_TABLE_ENTRY',
             'WRITE_NAME', 'READ_NAME',
             'READ_SERIAL',
             'WRITE_BRIGHTNESS', 'READ_BRIGHTNESS' ]

# Command format
# <report> <command> <data> * 31
for (cmd, name) in enumerate(COMMANDS):
    command = bytearray([0]*33)
    command[1] = cmd
    vars()[name] = command


class Led(object):
    """This class represents the state of a led in the plasmatrim.
    It operates like a tuple, but is mutable and has with extra
    functions."""

    __slots__ = 'red', 'green', 'blue'

    def __init__(self, *args):
        """Create a Led. Three required arguments; the value
        of red, green, and blue in RGB color space."""
        super(Led, self).__init__()
        for (key, val) in zip(self.__slots__, args):
            setattr(self, key, min(max(int(val), 0), 255))

    def __getitem__(self, index):
        """Get the values for the RGB sub-channels in that order."""
        return getattr(self, self.__slots__[index])

    def __setitem__(self, index, value):
        """Set the value for the RGB sub-channel in that order."""
        setattr(self, self.__slots__[index], max(0, min(255, value)))

    def __len__(self):
        """Three 8-bit sub-channels in 24-bit rgb."""
        return 3

    def __str__(self):
        """Print the Led's RGB in standard hex color format."""
        return "{:0>2X}{:0>2X}{:0>2X}".format(*self)

    def black(self):
        """Returns true if the Led is truly black."""
        return self.red == 0 and self.green == 0 and self.blue == 0

    def hsv(self, other):
        """Convert the RGB colorspace representation of this Led to
        HSV and return that as a tuple. Requires passing a target
        led color in order to avoid a singularity when RGB = 0,0,0."""
        if self.black():
            if other.black():
                return (0, 0, 0)
            else:
                (h, s, v) = other.hsv(self)
                return (h, s, 0)
        else:
            return rgb_to_hsv(self.red, self.green, self.blue)


class Lights(object):
    """Represent a whole plasmatrim's worth of Leds."""

    __slots__ = 'plasmatrim', 'leds'

    def __init__(self, plasmatrim):
        super(Lights, self).__init__()
        self.leds = [None] * len(self)
        self.plasmatrim = plasmatrim

    def __str__(self):
        """Pretty print a set of plasmatrim Leds"""
        leds = [str(led) for led in self]
        return "<Leds {}>".format(", ".join(leds))

    def hsv(self, other):
        """Convert the RGB representation of this plasmatrim to HSV and
        return that as a tuple. Requires passing the other state in
        order to avoid a singuarity error with a RGB = 0x000000."""
        pairs = zip(self, other)
        return [v for (s, o) in pairs for v in s.hsv(o)]

    def send(self, cmd):
        """Helper function to use the plasmatrim object to write to the
        physical device."""
        return self.plasmatrim._send(cmd)

    def show(self):
        """Makes the plasmatrim leds update to match this object."""
        cmd = IMMEDIATE_WRITE[:]
        # Double-loop to turn 8 leds with r,g,b each into a 24-byte
        # sequence which plasmatrim speaks.
        cmd[2:26] = [val for led in self for val in led]
        cmd[26] = self.plasmatrim.brightness
        self.send(cmd)

    def __len__(self):
        """PlasmaTrims have eight leds."""
        return 8

    def __getitem__(self, index):
        """Return the Led object of the index'th led."""
        return self.leds[index]

    def __setitem__(self, index, value):
        """Set the Led object of the index'th led."""
        self.leds[index] = value


class Current(Lights):
    """Current represents the current live state of the plasmatrim's
    led values. It auto-initializes from what is being shown on the
    plasmatrim."""
    def __init__(self, plasmatrim):
        super(Current, self).__init__(plasmatrim)
        leds = self.send(IMMEDIATE_READ)[:24]
        # Tricky, see the itertools docs for grouper
        self.leds = [Led(*rgb) for rgb in zip(*[iter(leds)]*3)]


class Slot(Lights):
    """Slot represents an entry in a program in the plasmatrim. It is
    an imprecice version of the live lights, where the stored values
    are only 4-bit instead of 8-bit.
    
    New properties:
        fade: how long to take fading between this and the next slot.
        hold: how long to wait in this state before the next transition.
    Hold and Fade are both measured in 1/8ths of a second."""

    __slots__ = 'hold', 'fade'

    def __init__(self, plasmatrim, slot):
        super(Slot, self).__init__(plasmatrim)
        (self.leds, self.hold, self.fade) = self.read(slot)

    def __str__(self):
        """Pretty print a set of plasmatrim Leds"""
        leds = [str(led) for led in self]
        return "<Leds {} {}/{}>".format(", ".join(leds), self.fade, self.hold)

    def read(self, slot):
        """Read the lights in the given slot, and convert them up into
        the normal 24-bit RGB color space. Reads the hold and fade
        transition parameteres too."""
        cmd = READ_TABLE_ENTRY[:]
        cmd[2] = slot
        #(*leds, holdfade) = self.send(cmd)[1:14]
        leds = self.send(cmd)[1:13]
        holdfade = self.send(cmd)[13]
        leds = [v * 16 for p in leds for v in [p >> 4, p & 0xF]]
        leds = [Led(*rgb) for rgb in zip(*[iter(leds)]*3)]
        return (leds, holdfade >> 4, holdfade & 0xF)

    def write(self, slot):
        """Write this set of lights to the plasmatrim in its slot.
        Down-converts from 24-bit RGB color space into 12-bit."""
        (leds, hold, fade) = self.read(slot)
        if leds != self.leds or hold != self.hold or fade != self.fade:
            cmd = WRITE_TABLE_ENTRY[:]
            cmd[2] = slot
            leds = [(val / 16) & 0xF for led in self for val in rgb]
            cmd[3:15] = [x << 4 + y for (x, y) in zip(*[iter(leds)]*2)]
            cmd[16] = (self.hold & 0xF) << 4 + self.fade & 0xF
            self.send(cmd)


class Program(object):
    """A sequence of slot objects which represent a program for the
    plasmatrim to cycle through."""

    def __init__(self, device):
        """Initialize a program from the plasmatrim itself."""
        self.device = device
        length = self.read_length()
        self.slots = [Slot(device, slot=i) for i in range(length)]

    def __str__(self):
        return "Program:\n  {}".format("\n  ".join(map(str, self.slots)))

    def __len__(self):
        return len(self.slots)

    def __getitem__(self, index):
        return self.slots[index]

    def __setitem__(self, index, value):
        self.slots[index] = value

    def __delitem__(self, index, value):
        del self.slots[index]

    def read_length(self):
        """Get the number of steps in the plasmatrim's program."""
        return self.device._send(READ_TABLE_LENGTH)[0]

    def write(self):
        """Write the program to the plasmatrim. Only actually writes
        to the plasmatrim's nvram if there have been changes to the
        length of the program, or if the slots at a given index
        differ."""
        if len(self) != self.read_length():
            cmd = WRITE_TABLE_LENGTH[:]
            cmd[1] = len(self)
            self.device._send(cmd)
        for (slot, leds) in enumerate(self.slots):
            leds.write(slot)

    def run(self, cycles=1, hertz=500):
        """Simulate what running the program would look like.
        
        Optional parameters:
            cycles: the number of times to loop through the sequence
            hertz: the number of steps per second to compute betwen
                   lighting states (the smoothness of the transition)
                   
        Having hertz too high is generally non-harmful on modern
        x86 hardware. The default value for hertz is 500, good on
        cPython 3.4 on an intel core i5-4200U."""
        seq = []
        shifted = self.slots[1:]
        shifted.append(self.slots[0])
        # Computes the iteration sequences
        for cur, tar in zip(self.slots, shifted):
            pairs = list(zip(cur.hsv(tar), tar.hsv(cur)))
            deltas = [(c, t - c) for (c, t) in pairs]
            # Correct deltas to take the shortest path through hsv
            # color space
            for i in range(0, len(deltas), 3):
                if deltas[i][-1] > 0.5:
                    deltas[i] = deltas[i][0] + 1, deltas[i][-1] - 1
                elif deltas[i][-1] < -0.5:
                    deltas[i] = deltas[i][0] - 1, deltas[i][-1] + 1
            # Pre-computing fade sequence.
            steps = int(hertz * tar.fade / 8)
            for step in range(steps):
                t = float(step) / steps
                at = [c + t * d for (c, d) in deltas]
                lights = Lights(self.device)
                for index, hsv in enumerate(zip(*[iter(at)]*3)):
                    lights[index] = Led(*hsv_to_rgb(*hsv))
                seq.append(lights)
            # Add duplicate hold entries to simplify run-time math.
            steps = int(hertz * tar.hold / 8)
            seq.extend([tar] * steps)
        # Run the show as fast as possible.
        start = time()
        now = start
        end = now + cycles * len(seq) / hertz
        while now < end:
            slot = int((now - start) * hertz) % len(seq)
            seq[slot].show()
            now = time()


class PlasmaTrim(object):
    def __init__(self, device):
        """Create a plasmatrim object which can be used to access the
        device specified."""
        super(PlasmaTrim, self).__init__()
        self._device = device
        self._fd = open(device, "w+b")
        self.name = self.read_name()
        self.brightness = self.read_brightness()
        self.leds = Current(self)
        self.program = Program(self)

    def read_name(self):
        """Read and return the name stored on the device."""
        return self._send(READ_NAME).decode().rstrip('\x00')

    def read_brightness(self):
        """Read and return the programmed default brightness."""
        return self._send(READ_BRIGHTNESS)[0]

    def write(self):
        """Write any changed device attributes to the plasmatrim."""
        if len(self.name) > 31:
            raise ValueError("Name can't be more than 31 bytes.")
        if self.brightness < 1 or self.brightness > 100:
            raise ValueError("Brightness must be between 1 and 100.")
        if self.name != self.read_name():
            cmd = WRITE_NAME[:]
            cmd[2:2+len(self.name)] = self.name
            self._send(cmd)
        if self.brightness != self.read_brightness():
            cmd = WRITE_BRIGHTNESS[:]
            cmd[2] = self.brightness
            self._send(cmd)

    @property
    def serial(self):
        """Read-only property containing the serial number."""
        if not hasattr(self, '_serial'):
            self._serial = struct.unpack("I", self._send(READ_SERIAL)[:4])[0]
        return self._serial

    def __del__(self):
        self._fd.close()

    def __str__(self):
        return "<{} {}>".format(self.name, self._device)

    def __repr__(self):
        return "PlasmaTrim('{}')".format(self._device)

    def start(self):
        """Start the plasmatrim itself running its program."""
        self._send(START_SEQUENCE)

    def stop(self):
        """Stop the plasmatrim itself running its program."""
        self._send(STOP_SEQUENCE)


    # FIXME
    # Rewrite this method to do better
    def _send(self, command, retries=5, timeout=100):
        fd = self._fd
        if len(command) != 33:
            raise ValueError("command must be 33 bytes long")
        handler = signal.signal(signal.SIGALRM, _TimeoutError.timeout)
        for attempt in range(retries):
            signal.setitimer(signal.ITIMER_REAL, timeout/1000.0)
            try:
                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("Write: {}", hexlify(command[1:]))
                fd.write(command)
                fd.flush()
                reply = bytearray(fd.read(32))
                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("Recv: {}", hexlify(reply))
                signal.setitimer(signal.ITIMER_REAL, 0)
                if reply[0] != command[1]:
                    msg = "Expected msg type {} but got {}"
                    raise IOError(msg.format(command[1], reply[0]))
                return reply[1:]
            except _TimeoutError:
                print("IO timed out, try #%d." % attempt)
                time.sleep(0.000001)
            finally:
                signal.signal(signal.SIGALRM, handler)
        msg = "Gving up on PlasmaTrim {}"
        raise IOError(msg.format(self))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Set plasmatrim LED colors.')
    parser.add_argument('--list', help='List all the plasmatrims',
                        action='store_true')
    parser.add_argument('--device', help='Apply to specified device')
    ex = parser.add_mutually_exclusive_group()
    ex.add_argument('--start', action='store_true',
                    help='Start program on plasmatrim.')
    ex.add_argument('--stop', action='store_true',
                    help='Stop program on plasmatrim.')
    parser.add_argument('--brightness', nargs=1, type=int,
                        help='Brightness percent')
    ex = parser.add_mutually_exclusive_group()
    ex.add_argument('--color', nargs=1, help='Color to set')
    ex.add_argument('--colors', dest='color', nargs=8,
                    help='Colors to set')
    ex.add_argument('--rgb', dest='rgb', nargs=3, type=int,
                    help='Color to set')
    ex.add_argument('--rgbs', dest='rgb', nargs=24, type=int,
                    help='Color to set')
    args = parser.parse_args()

    # Get the list to operate on
    devices = [PlasmaTrim(args.device)] if args.device else find()

    if args.list:
        for device in devices:
            print(device)
        sys.exit(0)
    elif args.start:
        for device in devices:
            device.start()
        sys.exit(0)
    elif args.stop:
        for device in devices:
            device.stop()
        sys.exit(0)

    if args.brightness:
        for device in devices:
            device.brightness = min(100, max(0, args.brightness[0]))

    if args.color:
        tuples = [colors.lookup(color) for color in args.color]
        args.rgb = [channel for rgb in tuples for channel in rgb]
    if args.rgb:
        if len(args.rgb) == 3:
            args.rgb = args.rgb * 8
        for device in devices:
            for i, led in enumerate(device.leds):
                for j, index in enumerate(led):
                    led[j] = args.rgb[i*len(led) + j]

    for device in devices:
        device.leds.show()
