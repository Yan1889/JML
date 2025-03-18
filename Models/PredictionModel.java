package libs.JML.Models;

import libs.JML.DataClasses.*;

import java.io.IOException;
import java.util.*;

/// Predicts a list of doubles based on an input list of doubles
public interface PredictionModel extends InferenceModel {

    /// trains the model on a dataset and returns the training time in milliseconds
    long trainOnDataset(Dataset dataset, int epochCount, boolean shouldShow);

    void trainOnPoint(List<Double> input, List<Double> output, boolean shouldShow);


    List<Double> predict(List<Double> input);

    double loss(List<Double> predicted, List<Double> observed);

    void writeToFile(String fileName) throws IOException;
}
