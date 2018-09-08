//--------------------------------------------------------------------------------------------------
//Date: 01-Sep-2018
//By: Ahmed Eissa
//Custom Naive Bayes Algorithm Model
//-------------------------------------------------------------------------------------------------
package org.apache.spark.ml.nu

import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.ml._
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types._

class EissaNaiveBayesModel(
                            override val uid: String,
                            data_mean: Array[Row],
                            data_var: Array[Row],
                            data_count: Array[Row],
                            rows_count :Long,
                            cols:scala.collection.mutable.ArrayBuffer[String],
                            NumberOfFeatures: Int)
  extends Model[EissaNaiveBayesModel] with EissaNaiveBayesParams with Serializable {



  override def copy(extra: ParamMap): EissaNaiveBayesModel = {
    defaultCopy(extra)
  }

  // check the given dataset Schema
  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  // predict the label for the given dataset
  override def transform(dataset: Dataset[_]): DataFrame = {
    var inputdata = dataset.toDF()
    val NumberOfClassess = data_count.length

    //--- calculate the probability of each class
    // this part is optional - not used to predict the label
    // i am just adding columns (one column for each Class) contains the
    // class probabilty, but i don't compare between them and select the highest probabilty
    //so, it is Optional or for Debuging only
    //----------------------------------------------------------------------
    val calc_prob = udf((data:String , num:Int )=>
    {
      var f = data.replace("[" , "").replace("]" , "")

      var temp = data_count(num)(1).toString().toDouble / rows_count
      for (j <- 0 to NumberOfFeatures-1) {
        val x = f.split(",")(j).toDouble
        temp = temp * FeatueProbability( x , data_mean(num)(j).toString().toDouble, data_var(num)(j).toString().toDouble)
      }
      temp

    })

    for ( i <-0 to NumberOfClassess - 1) {
      inputdata = inputdata.withColumn("Class_" + data_count(i)(0).toString() , calc_prob(inputdata("features").cast(org.apache.spark.sql.types.StringType), lit(i)))

    }


    // calculate the best probability calss
    val calc_pridiction = udf((data:String )=>
    {
      var f = data.replace("[" , "").replace("]" , "")
      var result: Double = 0
      var resultTmp: Double = 0.0
      var CurrentClass = "2"//data_count(0)(0).toString

      // Loop for each Class
      for ( i <-0 to NumberOfClassess - 1) {
        var resultTmp = data_count(i)(1).toString().toDouble / rows_count
        // looping for each feature in the Features Vector
        for (j <- 0 to NumberOfFeatures - 1) {
          val x = f.split(",")(j).toDouble
          // accumulate the probability for each Feature in the Feature Vector
          resultTmp = resultTmp * FeatueProbability(x, data_mean(i)(j).toString().toDouble, data_var(i)(j).toString().toDouble)
        }
        // Determine the class with the highest probability
        // by comare against temp value
        if (resultTmp > result) {
          result = resultTmp
          CurrentClass = data_count(i)(0).toString()
        }
        resultTmp = 0;
      }
      CurrentClass.toDouble
    })

    // call a user defined function for each Feature Column in each Row (after casting it to string)
    // this is the best way i found tell now to convert Vector dynamically to multiple columns
    // Vector( 1,3,5)  to col1=1, col2=3, col3=5 , i was not able to cast Vector and generate multiple column dynamically then added to the dataframe
    // directly, so i convert the vector to string, then did the calculation and return single number.
    inputdata = inputdata.withColumn("Prediciton" , calc_pridiction(inputdata("features").cast(org.apache.spark.sql.types.StringType)))

    //return the dataframe
    inputdata

  }


  // Calculate the Probability for each class for a spacific value
  def FeatueProbability(x: Double, mean_y: Double, variance_y: Double): Double = {
    var exponent = Math.exp((- Math.pow(x-mean_y,2)/(2 * variance_y)))
    return (1 / (math.sqrt(2 * Math.PI) * variance_y)) * exponent

  }

}




