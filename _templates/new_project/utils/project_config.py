from argparse import Namespace

class GlobalVar(Namespace):
    # Labels for classes
    LEFT = 1
    RIGHT = 0
    WRONG = -1

    # Colors for img files
    WHITE = 255
    BLACK = 0

    # Variables for plotting
    PADDING = 0.25
    DPI = 50

    # DATAOPTIONS
    train='train',
    vali='vali',
    test='test'
