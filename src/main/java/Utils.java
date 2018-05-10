import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.*;
import scala.Function1;

public class Utils
{

    public static void printMetrics(JavaPairRDD<Object, Object> predictionAndLabel)
    {
        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabel.rdd());
        double accuracy = metrics.accuracy();
        double precision = metrics.precision();
        double precision1 = metrics.precision(1);
        double recall = metrics.recall();
        double recall0 = metrics.recall(0);
        double recall1 = metrics.recall(1);

        System.out.println("Accuracy = " + accuracy);
        System.out.println("Precision = " + precision);
        System.out.println("Precision 1 = " + precision1);
        System.out.println("Recall = " + recall);
        System.out.println("Recall 0 = " + recall0);
        System.out.println("Recall 1 = " + recall1);

        // Confusion matrix
        Matrix confusion = metrics.confusionMatrix();
        System.out.println("Confusion matrix: \n" + confusion);

        // Stats by labels
        for (int i = 0; i < metrics.labels().length; i++)
        {
            System.out.format("Class %f precision = %f\n", metrics.labels()[i], metrics.precision(
                    metrics.labels()[i]));
            System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
                    metrics.labels()[i]));
            System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
                    metrics.labels()[i]));
        }
    }

    public static JavaRDD<LabeledPoint> loadAndParseData(JavaSparkContext sc, String path)
    {
        String datapath = path;

        JavaRDD<String> data = sc.textFile(datapath);

        JavaRDD<LabeledPoint> parsedTrainData = data
                .map(new Function<String, LabeledPoint>()
                {
                    public LabeledPoint call(String line) throws Exception
                    {
                        String[] parts = line.split(",");
                        return new LabeledPoint(Double.parseDouble(parts[5]),
                                Vectors.dense(Double.parseDouble(parts[0]),
                                        Double.parseDouble(parts[1]),
                                        Double.parseDouble(parts[2]),
                                        Double.parseDouble(parts[3]),
                                        Double.parseDouble(parts[4])));
                    }
                });

        return parsedTrainData;
    }

    public static void printMetrics(MulticlassMetrics metrics)
    {
        // Confusion matrix
        Matrix confusion = metrics.confusionMatrix();
        double accuracy = metrics.accuracy();
        double precision = metrics.precision();
        double recall = metrics.recall();

        System.out.println("Confusion matrix: \n" + confusion);
        System.out.println("Overall accuracy = " + accuracy);
        System.out.println("Overall precision = " + precision);
        System.out.println("Overall recall = " + recall);

        // Stats by labels
        for (int i = 0; i < metrics.labels().length; i++)
        {
            System.out.format("Class %f precision = %f\n", metrics.labels()[i], metrics.precision(
                    metrics.labels()[i]));
            System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
                    metrics.labels()[i]));
            System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
                    metrics.labels()[i]));
        }
    }
}
