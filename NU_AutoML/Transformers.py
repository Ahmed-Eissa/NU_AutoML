from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param

"""
------------------------------------------------------------------------------------------------------------------------
NU AutoML Aug 2018
Feature scaling is a method used to standardize the range of independent variables or features of data
it is also known as data normalization and is generally performed during the data preprocessing step.
------------------------------------------------------------------------------------------------------------------------
"""
class Rescaler(Transformer, HasInputCol, HasOutputCol):

	def __init__(self,inputCol , outputCol , StdType = 0):
		super(Rescaler, self).__init__()
		self.inputcol = inputCol
		self.outputcol = outputCol
		self.StdType = StdType
		

	"""
	execute the selected Type of Scaling on the spacified column
	
	"""
	def _transform(self, dataset):
		try:
			if dataset.select(self.inputcol).dtypes[0][1] == 'bigint' or \
				dataset.select(self.inputcol).dtypes[0][1] == 'int' or \
				dataset.select(self.inputcol).dtypes[0][1] == 'double' or \
				dataset.select(self.inputcol).dtypes[0][1] == 'float':
					maxval = dataset.agg({self.inputcol: "max"}).collect()[0][0]
					meanval = dataset.agg({self.inputcol: "mean"}).collect()[0][0]
					minval = dataset.agg({self.inputcol: "min"}).collect()[0][0]
					if( self.StdType == 0 ):
						dataset = dataset.withColumn(colName = self.outputcol , col=  (dataset[self.inputcol] - minval) /(maxval - minval)  )
					else:
						dataset = dataset.withColumn(colName=self.outputcol, col= (dataset[self.inputcol] - meanval) / (maxval - minval) )
					dataset = dataset.drop(self.inputcol)
		except Exception as ex:
			pass

		return dataset

"""
------------------------------------------------------------------------------------------------------------------------
In statistics, an outlier is an observation point that is distant from other observations
[Q1 - K * (Q3 - Q1)]  , [Q3 + K * (Q3 - Q1)] 
K default value will be 1.5
------------------------------------------------------------------------------------------------------------------------
"""
class OutlierRemover(Transformer, HasInputCol, HasOutputCol):

	def __init__(self,inputCol , outputCol , k = 1.5):
		super(OutlierRemover, self).__init__()
		self.inputcol = inputCol
		self.outputcol = outputCol
		self.k = k


	"""
    Remove Outlier rows by filter them out and return the dataset
    """
	def _transform(self, dataset):
		quantiles = dataset.stat.approxQuantile( self.inputcol , [0.25, 0.75], 0.0)
		Q1 = quantiles[0]
		Q3 = quantiles[1]
		IQR = Q3 - Q1
		lowerRange = Q1 - (self.k * IQR)
		upperRange = Q3 + (self.k * IQR)
		return dataset.filter( self.inputcol  + " >= " + str(lowerRange) + " and " + self.inputcol +" <= " + str(upperRange))


"""
------------------------------------------------------------------------------------------------------------------------
convert values to binary (0 or 1) based on threshold value
if value > threshold --> 1
if value < threshold --> 0
------------------------------------------------------------------------------------------------------------------------
"""
class binarizer(Transformer, HasInputCol, HasOutputCol):
	def __init__(self,inputCol , outputCol , threshold= 100):
		super(binarizer, self).__init__()
		self.inputcol = inputCol
		self.outputcol = outputCol
		self.threshold = threshold

	"""
	convert continous column to binary column
	"""
	def _transform(self, dataset):
		dataset = dataset.withColumn(colName=self.outputcol, col=( dataset[self.inputcol] > self.threshold).cast('integer') )
		dataset = dataset.drop(self.inputcol)
		return dataset
