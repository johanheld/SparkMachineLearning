import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

public class SVM {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("SVM").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        String path = "C:/Workspace/Github/BachelorData/Machine1_balanced_set.csv";

        JavaRDD<String> testData = sc.textFile(path);

        JavaRDD<LabeledPoint> parsedTrainData = testData
                .map(new Function<String, LabeledPoint>() {
                    public LabeledPoint call(String line) throws Exception {
                        String[] parts = line.split(",");
                        return new LabeledPoint(Double.parseDouble(parts[5]),
                                Vectors.dense(Double.parseDouble(parts[0]),
                                        Double.parseDouble(parts[1]),
                                        Double.parseDouble(parts[2]),
                                        Double.parseDouble(parts[3]),
                                        Double.parseDouble(parts[4])));
                    }
                });

       // JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD();

        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint> training = parsedTrainData.sample(false, 0.6, 11L);
        training.cache();
        JavaRDD<LabeledPoint> test = parsedTrainData.subtract(training);

        // Run training algorithm to build the model.
        int numIterations = 1000;
        SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

        // Clear the default threshold.
        model.clearThreshold();

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));

        // Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        JavaRDD<Tuple2<Object, Object>> pr = metrics.pr().toJavaRDD();

        // Recall by threshold
        JavaRDD<Tuple2<Object, Object>> recall = metrics.recallByThreshold().toJavaRDD();

        // F Score by threshold
        JavaRDD<Tuple2<Object, Object>> f1Score = metrics.fMeasureByThreshold().toJavaRDD();

        JavaRDD<Tuple2<Object, Object>> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();

        // Precision-recall curve
        JavaRDD<Tuple2<Object, Object>> prc = metrics.pr().toJavaRDD();

        // Thresholds
        JavaRDD<Double> thresholds = precision.map(
                new Function<Tuple2<Object, Object>, Double>() {
                    @Override
                    public Double call(Tuple2<Object, Object> t) {
                        return new Double(t._1().toString());
                    }
                }
        );

        // ROC Curve
        JavaRDD<Tuple2<Object, Object>> roc = metrics.roc().toJavaRDD();


        System.out.println("ROC curve: " + roc.collect());
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());
        System.out.println("Precision by threshold: " + precision.collect());
        System.out.println("Area under ROC = " + metrics.areaUnderROC());
        System.out.println("Precision-recall curve: " + prc.collect());
        System.out.println("F2 Score by threshold: " + f2Score.collect());
        System.out.println("Recall by threshold: " + recall.collect());
        System.out.println("F1 Score by threshold: " + f1Score.collect());
        System.out.println("PR: " + pr.collect());



    }
}