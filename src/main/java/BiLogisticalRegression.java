import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;

/**
 * Created by johan on 2018-05-07.
 */
public class BiLogisticalRegression
{
    public static void main(String[] args)
    {
        SparkConf conf = new SparkConf().setAppName("JavaLogisticRegressionWithLBFGSExample").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);

        String trainPath = "file:///C:/Users/johan/Desktop/data/Machine1_balanced_set.csv";
//        String testPath = "file:///C:/Users/johan/Desktop/data/Machine1_noDates.csv";

        JavaRDD<LabeledPoint> parsedTrainData = Utils.loadAndParseData(sc, trainPath);

        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint>[] splits = parsedTrainData.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];

        // Run training algorithm to build the model.
        LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(2) //Ändrat från 10 till 2 här
                .run(training.rdd());

        // Compute raw scores on the test set.
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        Utils.printMetrics(metrics);
        sc.stop();
    }
}
