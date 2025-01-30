import math
import os
import time


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def draw_sinusoidal(amplitude, frequency, speed):
    width = 80
    height = 24

    while True:
        clear_screen()
        for y in range(height):
            for x in range(width):
                sin_value = amplitude * math.sin(2 * math.pi * frequency * (y + time.time() * speed))
                sin_pos = int((sin_value + amplitude) * (width - 1) / (2 * amplitude))
                if x == sin_pos:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        time.sleep(0.1)


amplitude = 1
frequency = 0.05
speed = 5

draw_sinusoidal(amplitude, frequency, speed)
