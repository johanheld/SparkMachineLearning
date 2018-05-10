import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

public class NaiveBayes
{

    public static void main(String[] args)
    {
        SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String trainPath = "file:///C:/Users/johan/Desktop/data/all_machines_balanced.csv";
        JavaRDD<LabeledPoint> parsedTrainData = Utils.loadAndParseData(jsc, trainPath);
        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = parsedTrainData.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        NaiveBayesModel model = org.apache.spark.mllib.classification.NaiveBayes.train(trainingData.rdd(), 1.0);
        JavaPairRDD<Object, Object> predictionAndLabel =
                testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabel.rdd());
        Utils.printMetrics(metrics);
        jsc.stop();
    }
}
