//---------------------------------------------------------------------------------------------------//
//---- Date: 26 Oct 2018
//---- By : Ahmed Eissa
//---- Description: Grid Search for Random forest hyper parameters
//---- Dataset: Cover Type [ https://archive.ics.uci.edu/ml/datasets/SUSY ]
//---------------------------------------------------------------------------------------------------//
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml._
import org.joda.time.DateTime

object SUSYDatasetApp {


  def main (arg: Array[String]): Unit = {

    print("----------------------------start------------------------------------------- " + "\n")
    //Create Spark Session
    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()
    print("Session Created " + "\n")

    // Load Data
    var training = spark.read.option("header","true").option("inferSchema","true").format("csv").load("wasb:///example/data/training_data.csv")
    var testing  = spark.read.option("header","true").option("inferSchema","true").format("csv").load("wasb:///example/data/testing_data.csv")//("/home/eissa/mycode/data/testing_data.csv")
    print("Data Laoded " +  "\n")

    // Process data
    val featureCols = Array("c1","c2" ,"c3","c4","c5","c6","c7","c8",
    "c9","c10","c11","c12","c13","c14","c15",
    "c16","c17","c18")

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val training_data = assembler.transform(training)
    val testing_data = assembler.transform(testing)
    print("Data Processed " + "\n")

    // Hyper Parameters
    val NumTree = Array(10, 50, 100,  150, 200)
    val MaxDepth = Array( 5, 10, 15 , 20 )
    val FeatureSubset = Array("all", "sqrt", "log2" , "onethird")
    val Impurity = Array("gini", "entropy")
    val MaxBins = Array(10, 100, 200 , 300)
    var count: Int = 0


    //Grid Search
    for(i <- 0 until NumTree.length)
    {
      for(j <- 0 until MaxDepth.length)
      {
        for(k <- 0 until FeatureSubset.length)
        {
          for(l <- 0 until Impurity.length)
          {
            for(m <- 0 until MaxBins.length)
            {
              count = count + 1;
              if ( count > 198) {
                val reuslt = Train_RandomForest(count, NumTree(i), MaxDepth(j), FeatureSubset(k), MaxBins(m), Impurity(l), training_data, testing_data)
                print(count + "," + NumTree(i) + "," + MaxDepth(j) + "," + FeatureSubset(k) + "," + Impurity(l) + "," + MaxBins(m) + "," + reuslt._2 + "," + reuslt._3 + "\n")
              }
            }
          }
        }
      }
    }
    print("----------------------------End-------------------------------------------")
  }


  // Random Forest Algorithm Training
  def Train_RandomForest(i: Int , NumTree: Int , MaxDepth: Int, FeatureSubset: String , MaxBins: Int , Impurity: String ,  training_data: DataFrame , testing_data: DataFrame ):  Tuple3[Int , Double,Double] = {

    // Set Start Data
    val t1 = DateTime.now()

    // Build the Classifer
    val rf = new RandomForestClassifier()
      .setLabelCol("Label")
      .setFeaturesCol("features")
      .setImpurity(Impurity)
      .setMaxBins(MaxBins)
      .setNumTrees(NumTree)
      .setFeatureSubsetStrategy(FeatureSubset)
      .setMaxDepth(MaxDepth)

    // Create pipeline & Train
    val pipeline = new Pipeline()
      .setStages(Array(rf))
    val model = pipeline.fit(training_data)

    // Set End Data
    val t2 = DateTime.now()
    val diffInMillis = t2.getMillis() - t1.getMillis();



    // Select (prediction, true label) and compute test error
    //------------------------------------------------------------------------------
    val predictions = model.transform(testing_data)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions) * 100
    return(i , accuracy,diffInMillis/1000.0)
  }


}
