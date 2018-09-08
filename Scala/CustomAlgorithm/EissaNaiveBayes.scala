
//--------------------------------------------------------------------------------------------------
//Date: 01-Sep-2018
//By: Ahmed Eissa
//Custom Estimator for (Naive Bayes Algorithm)
//-------------------------------------------------------------------------------------------------
package org.apache.spark.ml.nu

import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.ml._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.lit


class EissaNaiveBayes (override val uid: String , sc: org.apache.spark.SparkContext) extends Estimator[EissaNaiveBayesModel] with EissaNaiveBayesParams {
  // the name of the Feature Column
  def setInputCol(value: String) = set(inputCol, value)

  // the name of the Label Column
  def setTargetCol(value: String) = set(targetCol, value)



  override def copy(extra: ParamMap): EissaNaiveBayes = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  // Group dataframe dynamically by creating a map of  the column & aggregation function
  def groupAndAggregate(df: DataFrame,  aggregateFun: Map[String, String], cols: List[String] ): DataFrame ={
    val grouped = df.groupBy(cols.head, cols.tail: _*)
    val aggregated = grouped.agg(aggregateFun)
    aggregated
  }

  override def fit(dataset: Dataset[_]): EissaNaiveBayesModel = {

    var inputdata = dataset.toDF()
    // get number of features
    var NumberOfFeatures = inputdata.select($(inputCol)).first().getAs[Vector](0).toString().count(_ == ',') + 1
    // Empty ArrayBuffer to contains Features Column names (Feature_0  -> Feature_n)
    var cols: scala.collection.mutable.ArrayBuffer[String] =  scala.collection.mutable.ArrayBuffer("")

    // convert feature Vector to Columns (to calculate mean and variance)
    // appen columns to the inputdata
    val replace = udf((data:String , num:Int )=>

    { if ((data.replace("[" , "").replace("]" , "").split(",")(num) == "NaN" ) || (data.replace("[" , "").replace("]" , "").split(",")(num) == "" ))
      0.0
    else
      data.replace("[" , "").replace("]" , "").split(",")(num).toDouble
    })

    for ( i <-0 to NumberOfFeatures - 1) {
      inputdata= inputdata.withColumn( "Feature_" + i, replace(inputdata($(inputCol)).cast(org.apache.spark.sql.types.StringType), lit(i)))
      cols += ("Feature_" + i.toString())
    }
    cols.remove(0)
    //------------------------------------------------------------------------

    //-- calculate variance (keep column order)
    var cols_var_after = cols.map { x => "var_samp(" +x+ ")" }
    var m1 = Map( cols(0) -> "var_samp")
    for (p <- 1 to cols.length - 1)  {   m1 = m1 +  ( cols(p) -> "var_samp")   }
    val data_var = groupAndAggregate( inputdata ,  m1 , List($(targetCol))).select(cols_var_after.head, cols_var_after.tail: _*).collect()
    //data_var.show(5)
    //-- calculate mean (keep column order)
    var cols_mean_after = cols.map { x => "avg(" +x+ ")" }
    var m2 = Map( cols(0) -> "MEAN")
    for (p <- 1 to cols.length - 1)  {   m2 = m2 +  ( cols(p) -> "MEAN")   }
    val data_mean = groupAndAggregate( inputdata ,  m2 , List($(targetCol)) ).select(cols_mean_after.head, cols_mean_after.tail: _*).collect()
    //data_mean.show()

    // calculate number of record per class
    val data_count = inputdata.groupBy($(targetCol)).count().collect()
    // get number of classes
    val NumberOfClassess = data_count.length
    // get Total Number of rows
    val rows_count = inputdata.select($(targetCol)).count()

    // Calling the model and sending the calculated info (traning output)
    new EissaNaiveBayesModel(uid,  data_mean , data_var , data_count ,rows_count ,  cols  , NumberOfFeatures)
  }
}

