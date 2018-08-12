from pyspark.sql.functions import desc

"""
=======================================================================================================================
NU AutoML Augest 2018
SimpleKNN Machine LEarning Algorithm
it use euclidean Distance to get the nearest neighbor points.
=======================================================================================================================
"""
class SimpleKNN:
    def __init__(self , dataset ,  labelcol , k=3 ):
        self.dataset = dataset
        self.labelcol = labelcol
        self.k = k


    """
    Calcualte the distance between certain instance
    and the given dataset.
    """
    def ClacuateDistanc(self , y):
        isfirst = 0
        cols = self.dataset.columns
        cols.remove(self.labelcol)
        for i in cols:
            if (isfirst == 0):
                result = (self.dataset[i] - y[i]) ** 2
                isfirst = 1
            else:
                result = result + (self.dataset[i] - y[i]) ** 2
        return result ** (1 / 2)

    """
    do the voting to determine which class the instance belong to.
    """
    def getNearestClass(self):
        sorteddf = self.dataset.sort(desc("distances")).limit(self.k).groupBy(self.labelcol).count()
        result = sortesddf.sort(desc("count")).limit(1).collect()
        return result[0][0]

    """
    Classify given instance
    """
    def getKNN(self, point):
        self.dataset = self.dataset.withColumn('distances', self.ClacuateDistanc(point))
        return self.getNearestClass()



