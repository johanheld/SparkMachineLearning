import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.*;
import scala.Function1;
import scala.Tuple2;

import static org.apache.spark.sql.types.DataTypes.DoubleType;

public class MultilayerPerceptron {

   public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf().setAppName("GBT").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        SparkSession spark = new SparkSession(jsc.sc());

        Dataset<Row> csv = spark.read().format("csv").option("header","false").load("C:/Workspace/Github/BachelorData/Machine1_balanced_set.csv");

       Dataset<Row> analysisData  = csv.withColumn("_c0", csv.col("_c0").cast(DoubleType));
       analysisData  = analysisData.withColumn("_c1", csv.col("_c1").cast(DoubleType));
       analysisData  = analysisData.withColumn("_c2", csv.col("_c2").cast(DoubleType));
       analysisData  = analysisData.withColumn("_c3", csv.col("_c3").cast(DoubleType));
       analysisData  = analysisData.withColumn("_c4", csv.col("_c4").cast(DoubleType));
       analysisData  = analysisData.withColumn("_c5", csv.col("_c5").cast(DoubleType));

        VectorAssembler va = new VectorAssembler();
        String[] featureCols = {"_c0","_c1","_c2","_c3","_c4"};
        va.setInputCols(featureCols);
        va.setOutputCol("features");


        Dataset<Row> ds = va.transform(analysisData);
        Dataset<Row> preparedDs = ds.drop("_c0").drop("_c1").drop("_c2").drop("_c3").drop("_c4");
        preparedDs.show();

        Dataset<Row>[] splits = preparedDs.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        // specify layers for the neural network:
        // input layer of size 4 (features), two intermediate of size 5 and 4
        // and output of size 3 (classes)
        int[] layers = new int[] {5, 5, 4, 3};

        // create the trainer and set its parameters
        MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setFeaturesCol("features")
                .setLabelCol("_c5")
                .setMaxIter(100);


        // train the model
        MultilayerPerceptronClassificationModel model = trainer.fit(train);
       // compute accuracy on the test set
       Dataset<Row> result = model.transform(test);

       Dataset<Row> predictionAndLabels = result.select("prediction", trainer.getLabelCol());

       MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
               .setMetricName("accuracy");
       evaluator.setLabelCol("_c5");
       System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));

       MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels);
       Utils.printMetrics(metrics);

       spark.stop();
    }

  /*
  public static void main(String[] args) {
      SparkConf sparkConf = new SparkConf().setAppName("GBT").setMaster("local");
      JavaSparkContext jsc = new JavaSparkContext(sparkConf);
      SparkSession spark = new SparkSession(jsc.sc());
      JavaRDD<LabeledPoint> parsedTrainData =  Utils.loadAndParseData(jsc);
      Dataset<Row> dataFrame = spark.createDataFrame(parsedTrainData.rdd(),LabeledPoint.class);

     // Split the data into train and test
      Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
      Dataset<Row> train = splits[0];
      Dataset<Row> test = splits[1];

      // specify layers for the neural network:
      // input layer of size 4 (features), two intermediate of size 5 and 4
      // and output of size 3 (classes)
      int[] layers = new int[] {4, 5, 4, 3};

      // create the trainer and set its parameters
      MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
              .setLayers(layers)
              .setBlockSize(128)
              .setSeed(1234L)
              .setMaxIter(100);

      // train the model
      MultilayerPerceptronClassificationModel model = trainer.fit(train);

      // compute accuracy on the test set
      Dataset<Row> result = model.transform(test);
      Dataset<Row> predictionAndLabels = result.select("prediction", "label");
      MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
              .setMetricName("accuracy");

      System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
      // $example off$

      spark.stop();
  }*/
}
