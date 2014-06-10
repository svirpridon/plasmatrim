"""A module which looks up /etc/X11/rgb.txt to provide color to rgb
value translations.

Lookup file may be overridden by changing the module's colors variable
to point to an alternate locaiton."""

__all__ = ['lookup']

import sys

FILENAME = '/etc/X11/rgb.txt'
COLORS = dict()

def lookup(color):
    """Look up a color in the colors database. Raises KeyError if the
    color is unknown.
    
    >>> lookup('red')
    (255, 0, 0)
    >>> lookup('mint cream')
    (245, 255, 250)
    """
    if len(COLORS) == 0:
        with open(FILENAME) as source:
            for line in source:
                if line.startswith('!'):
                    continue
                line = line.strip()
                (r, g, b, name) = line.split(None, 3)
                COLORS[name] = int(r), int(g), int(b)
    return COLORS[color]
