/**
 * Created by johan on 2018-05-08.
 */

// $example on$

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;

public class DecisionTreeClassifier
{
    public static void main(String[] args)
    {

        SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTreeClassificationExample").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        String trainPath = "file:///C:/Users/johan/Desktop/data/all_machines_balanced.csv";
//        String testPath = "file:///C:/Users/johan/Desktop/data/all_machines_balanced.csv";

        JavaRDD<LabeledPoint> parsedTrainData = Utils.loadAndParseData(sc, trainPath);

        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = parsedTrainData.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> training = splits[0];
        JavaRDD<LabeledPoint> test = splits[1];

        int numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "gini";
        int maxDepth = 5;
        int maxBins = 32;

        // Train a DecisionTree model for classification.
        DecisionTreeModel model = DecisionTree.trainClassifier(training, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Object, Object> predictionAndLabel =
                test.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        double testErr =
                predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) test.count();

        System.out.println("Test Error: " + testErr);
        System.out.println("Learned classification tree model:\n" + model.toDebugString());

        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabel.rdd());
        Utils.printMetrics(metrics);
        sc.stop();
    }
}
