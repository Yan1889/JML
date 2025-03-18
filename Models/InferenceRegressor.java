package libs.JML.Models;

import libs.JML.Exceptions.DimensionalityException;
import libs.JML.util.ActivationFunction;
import libs.JML.Exceptions.ActivationFunctionException;

import java.util.ArrayList;
import java.util.List;

public class InferenceRegressor implements InferenceModel {

    // parameters
    public final List<List<List<Double>>> weights;
    public final List<List<Double>> biases;

    // hyper parameters
    public final List<Integer> layerSizes;
    public final List<ActivationFunction> activationFunctions;

    public InferenceRegressor(Regressor source) {
        layerSizes = source.layerSizes;
        activationFunctions = source.activationFunctions;


        weights = new ArrayList<>();

        for (var  weightLayer : source.weights) {
            List<List<Double>> weightLayer_copy = new ArrayList<>();

            for (var weightThing : weightLayer) {
                weightLayer_copy.add(new ArrayList<>(weightThing));
            }

            weights.add(weightLayer_copy);
        }

        biases = new ArrayList<>();

        for (var biasLayer : source.biases) {
            biases.add(new ArrayList<>(biasLayer));
        }
    }

    @Override
    public double loss(List<Double> predicted, List<Double> observed) {
        if (layerSizes.getLast() != predicted.size() || predicted.size() != observed.size()) throw new DimensionalityException();

        // (y - a)^2
        double error = 0;
        for (int i = 0; i < predicted.size(); i++) {
            double diff = observed.get(i) - predicted.get(i);
            error += diff * diff;
        }
        return error;
    }

    @Override
    public List<Double> predict(List<Double> input) {
        List<List<Double>> activations = new ArrayList<>();

        for (int l = 0; l < layerSizes.size(); l++) {
            List<Double> zValuesOfCurrentLayer = l > 0 ? passOnceAndGetZValues(l, activations) : input;

            activations.add(
                    applyActivationFunctionToLayer(
                            zValuesOfCurrentLayer,
                            activationFunctions.get(l)
                    )
            );
        }

        return activations.getLast();
    }


    private List<Double> passOnceAndGetZValues(int l, List<List<Double>> activations) {
        List<Double> zValuesOfCurrentLayer = new ArrayList<>();
        for (int i_curr = 0; i_curr < layerSizes.get(l); i_curr++) {
            double sum = 0;
            for (int i_prev = 0; i_prev < layerSizes.get(l - 1); i_prev++) {
                double activation_prev = activations.get(l - 1).get(i_prev);
                double weight_connecting = weights.get(l - 1).get(i_prev).get(i_curr);
                sum += activation_prev * weight_connecting;
            }
            double bias = biases.get(l).get(i_curr);
            zValuesOfCurrentLayer.add(sum + bias);
        }
        return zValuesOfCurrentLayer;
    }

    private List<Double> applyActivationFunctionToLayer(List<Double> layer, ActivationFunction functionType) {
        return layer.stream().map(e -> af(e, functionType)).toList();
    }

    private double af(double a, ActivationFunction functionType) {
        return switch (functionType) {
            case Identity -> a;
            case ReLU -> a > 0 ? a : 0;
            case Sigmoid -> 1 / (1 + Math.exp(-a));
            case Tanh -> Math.tanh(a);
            default -> throw new ActivationFunctionException("Use of not implemented activation function: " + functionType);
        };
    }
}
