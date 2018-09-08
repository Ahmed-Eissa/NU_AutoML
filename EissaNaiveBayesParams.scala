
//--------------------------------------------------------------------------------------------------
//Date: 01-Sep-2018
//By: Ahmed Eissa
//Testing Custom Parameters for Custom Naive Bayes Algorithm
//-------------------------------------------------------------------------------------------------


package org.apache.spark.ml.nu

import org.apache.spark.ml.param._

trait EissaNaiveBayesParams extends Params {
  final val inputCol= new Param[String](this, "inputCol", "The input column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")
  final val targetCol= new Param[String](this, "targetCol", "The target column")

}