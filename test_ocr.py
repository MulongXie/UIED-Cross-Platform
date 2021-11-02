from match_elements.GUI_pair import GUIPair
import cv2
from matplotlib import pyplot as plt
from random import randint as rint
from difflib import SequenceMatcher

gui = GUIPair('0002')

gui.element_detection(True, False, False)