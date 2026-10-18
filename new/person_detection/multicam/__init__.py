"""Several cameras, one set of people.

floor.py  maps a point in a camera picture to a spot on the floor (metres),
          from 4+ floor marks the admin clicks once per camera.
hub.py    joins the people seen by different cameras into one "global"
          person (G1, G2 ...) when they stand on the same spot at the same
          time, and shares a name recognised by one camera with the others.
"""
