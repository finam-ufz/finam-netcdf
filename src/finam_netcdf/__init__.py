class Layer:
    """
    Defines a NetCDF layer (2D data array).

    :param var: layer variable
    :param x: x coordinate variable
    :param y: y coordinate variable
    :param fixed: dictionary for further, fixed index coordinate variables (e.g. 'time')
    """

    def __init__(self, var: str, x: str, y: str, fixed: dict = {}):
        self.var = var
        self.x = x
        self.y = y
        self.fixed = fixed
