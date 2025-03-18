package libs.JML.Models;

import java.util.List;

/// Predicts a list of doubles based on an input list of doubles.
/// Cannot be changed.
public interface InferenceModel {
    List<Double> predict(List<Double> input);
    double loss(List<Double> predicted, List<Double> observed);
}
