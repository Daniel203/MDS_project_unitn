MARK_SIZE = 1024
"""The size of the watermark. It's 1024 as defined in the challange constraints"""

ALPHA = 1.0
"""The alpha value used when embedding and decoding"""

THRESH = 0.06
"""Threshold value used to determine if a detected watermark is valid or not"""

INPUT_DIR = "input"
"""Folder containing images to use"""

OUTPUT_DIR = "output"
"""Folder where embedded images will be saved"""

WATERMARK_NAME = "watermark.npy"
"""Name of the watermark file"""

MID_FREQ_START = 5000
"""Skip the first MID_FREQ_START frequencies when embedding and detecting"""
