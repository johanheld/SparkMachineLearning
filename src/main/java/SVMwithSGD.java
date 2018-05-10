/**
 * Created by johan on 2018-05-08.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

// $example on$
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

public class SVMwithSGD
{


    public static void main(String[] args)
    {
        SparkConf conf = new SparkConf().setAppName("JavaSVMWithSGDExample").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        // $example on$

        String trainPath = "file:///C:/Users/johan/Desktop/data/Machine1_balanced_set.csv";
//            String testPath = "file:///C:/Users/johan/Desktop/data/Machine1_noDates.csv";

        JavaRDD<String> trainData = sc.textFile(trainPath);
//            JavaRDD<String> testData = sc.textFile(testPath);

        JavaRDD<LabeledPoint> parsedTrainData = trainData
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


        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint> training = parsedTrainData.sample(false, 0.6, 11L);
        training.cache();
        JavaRDD<LabeledPoint> test = parsedTrainData.subtract(training);

        // Run training algorithm to build the model.
        int numIterations = 100;
        SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

        // Clear the default threshold.
        model.clearThreshold();

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();

        System.out.println("Area under ROC = " + auROC);
        // Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        System.out.println("Precision by threshold: " + precision.collect());

// Recall by threshold
        JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
        System.out.println("Recall by threshold: " + recall.collect());

// F Score by threshold
        JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
        System.out.println("F1 Score by threshold: " + f1Score.collect());

        JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
        System.out.println("F2 Score by threshold: " + f2Score.collect());

// Precision-recall curve
        JavaRDD<?> prc = metrics.pr().toJavaRDD();
        System.out.println("Precision-recall curve: " + prc.collect());

// Thresholds
        JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));

// ROC Curve
        JavaRDD<?> roc = metrics.roc().toJavaRDD();
        System.out.println("ROC curve: " + roc.collect());

// AUPRC
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());

// AUROC
        System.out.println("Area under ROC = " + metrics.areaUnderROC());


        // Save and load model
//        model.save(sc, "target/tmp/javaSVMWithSGDModel");
//        SVMModel sameModel = SVMModel.load(sc, "target/tmp/javaSVMWithSGDModel");
        // $example off$

        sc.stop();
    }


}
