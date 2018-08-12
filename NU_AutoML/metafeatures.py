

"""
1- get number of features
2- get number of instances
3- get number of continous features
4- get number of categorical column
5- number of classes
6- Target Entropy
7- Number of Features with Low Entropy ( 0.0 -> 0.33 )
8- Number of Features with Medium Entropy ( 3.4 -> 0.66)
9- Number of Feature with High Entropy ( 0.67 -> 1)
10 - Number of Features with Low skewness
"""
class caps:
    def __init__(self, s):
        self.s = s;

    def CapLetters(self):
        return self.s.upper()
