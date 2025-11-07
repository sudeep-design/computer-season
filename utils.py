# utils.py
import math
import numpy as np

def calculate_angle(a, b, c):
    """
    angle at point b formed by points a-b-c (a,b,c are (x,y) tuples or arrays)
    returns angle in degrees in [0,180]
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def landmark_to_point(landmark, width, height):
    """Convert a mediapipe normalized landmark to pixel coords (x,y)."""
    return (int(landmark.x * width), int(landmark.y * height))
