class HeaderIsInvalid(Exception):
    def __init__(self):
        self.message = 'VW header is invalid!'

class NotEnoughYLabelsInYForOAA(Exception):
    def __init__(self):
        self.message = 'Not enough y labels in y array for current oaa setting!'