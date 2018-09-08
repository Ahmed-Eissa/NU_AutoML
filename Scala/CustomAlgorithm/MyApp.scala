//--------------------------------------------------------------------------------------------------
//Date: 8-Sep-2018
//By: Ahmed Eissa
//Testing Custom Naive Bayes Algorithm
//-------------------------------------------------------------------------------------------------
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml._
import org.apache.spark.ml.nu._
object MyApp {

  def main(args: Array[String]): Unit = {
    //Create Spark Session
    val spark = SparkSession
      .builder()
      .appName("Java Spark SQL basic example")
      .config("spark.master", "local")
      .getOrCreate();
    // Load Testing Data (i am using "customer_churn.csv")
    val rawdata = spark.read.option("header","true").option("inferSchema","true").format("csv").load("/home/eissa/mycode/data/customer_churn.csv")
    // i am selecting important features
    val data = rawdata.select( "Churn" ,"Age","Total_Purchase","Account_Manager", "Years" , "Num_Sites" )

    //define the feature columns to put them in the feature vector**
    val featureCols = Array("Age","Total_Purchase","Account_Manager", "Years" , "Num_Sites" )
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    var Alldata = assembler.transform(data)

    //split Data into training and Testing
    val sets = Alldata.randomSplit(Array[Double](0.7, 0.3), 18)
    val training = sets(0)
    val test = sets(1)

    //set the input and output column names**
    val nb = new EissaNaiveBayes("demo" , spark.sparkContext)
    nb.setInputCol("features")
    nb.setTargetCol("Churn")
    // Create Pipeline
    val pipeline = new Pipeline().setStages(Array(nb))
    //Train
    val model = pipeline.fit(training)
    //Predict on Test Data
    val results = model.transform(test)

    // Convert the test results to an RDD of "predictionAndLabels"  using .as and .rdd
    val sqlContext = new org.apache.spark.sql.SQLContext(spark.sparkContext)
    import sqlContext.implicits._
    val predictionAndLabels = results.select("Prediciton","Churn").as[(Double, Double)].rdd

    // Instantiate a new MulticlassMetrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Print out the Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    //print Accuracy
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l => println(s"Precision($l) = " + metrics.precision(l))
    }
    spark.stop()
  }

}
