package libs.JML.Models;

import java.io.*;
import java.util.*;

import libs.JML.Exceptions.*;
import libs.JML.DataClasses.*;
import libs.JML.util.ActivationFunction;
import libs.JML.util.NNState;

public class Regressor implements PredictionModel, Serializable {

    // parameters
    public final List<List<List<Double>>> weights;
    public final List<List<Double>> biases;

    // hyper parameters
    public final List<Integer> layerSizes;
    public final List<ActivationFunction> activationFunctions;
    public final double learningRate;


    public int epochCount = 0;

    /// Creates a Regressor with the specified dimensions and ReLU as a default activation function for all hidden layers as well as the output layer.
    public Regressor(List<Integer> layerSizes, double learningRate) {
        this(layerSizes, constructListWithNElements(ActivationFunction.ReLU, layerSizes.size() - 2), learningRate);
    }

    /// Creates a Regressor with the specified dimensions and activation functions. Internally an identity activation function gets added for the input layer.
    public Regressor(List<Integer> layerSizes, List<ActivationFunction> activationFunctions, double learningRate) {
        this.layerSizes = layerSizes;
        this.learningRate = learningRate;

        this.activationFunctions = new ArrayList<>();
        this.activationFunctions.add(ActivationFunction.Identity);
        this.activationFunctions.addAll(activationFunctions);
        this.activationFunctions.add(ActivationFunction.Identity);

        if (this.activationFunctions.size() != layerSizes.size()) throw new ActivationFunctionException("Wrong amount of activation functions provided");

        this.biases = getRandomBiases();
        this.weights = getRandomWeights();
    }

    private static <T> List<T> constructListWithNElements(T defaultObject, int n) {
        List<T> ret = new ArrayList<T>();
        for (int i = 0; i < n; i++) {
            ret.add(defaultObject);
        }
        return ret;
    }

    @Override
    public long trainOnDataset(Dataset dataset, int epochCount, boolean shouldShow) {
        if (dataset.getDimensionX() != layerSizes.getFirst() || dataset.getDimensionY() != layerSizes.getLast()) throw new DimensionalityException("Dataset has wrong dimensionality");

        long startTime = System.currentTimeMillis();

        for (int i = 0; i < epochCount; i++) {
            double totalLoss = 0;
            for (DataPoint dataPoint : dataset.getDataPoints()) {
                trainOnPoint(dataPoint.x(), dataPoint.y(), shouldShow);
                if (shouldShow) {
                    totalLoss += loss(predict(dataPoint.x()), dataPoint.y());
                }
            }
            this.epochCount += 1;

            // Collections.shuffle(dataset.getDataPoints());

            if (shouldShow) {
                System.out.println("----");
                System.out.println("Total epoch loss: " + totalLoss);
                System.out.println("------------");
            }
        }


        return System.currentTimeMillis() - startTime;
    }

    @Override
    public void trainOnPoint(List<Double> input, List<Double> output, boolean shouldShow) {
        NNState currentState = feedForward(input);

        if (shouldShow) {
            System.out.println("Loss for" + input + ": " + loss(currentState.activations.getLast(), output));
        }

        updateParams(currentState, output);
    }

    @Override
    public List<Double> predict(List<Double> input) {
        return feedForward(input).activations.getLast();
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

    private NNState feedForward(List<Double> input) {
        NNState state = new NNState();

        for (int l = 0; l < layerSizes.size(); l++) {
            List<Double> zValuesOfCurrentLayer = l > 0 ? passOnceAndGetZValues(l, state) : input;

            state.zValues.add(zValuesOfCurrentLayer);
            state.activations.add(
                    applyActivationFunctionToLayer(
                            zValuesOfCurrentLayer,
                            activationFunctions.get(l)
                    )
            );
        }

        return state;
    }

    private List<Double> passOnceAndGetZValues(int l, NNState state) {
        List<Double> zValuesOfCurrentLayer = new ArrayList<>();
        for (int i_curr = 0; i_curr < layerSizes.get(l); i_curr++) {
            double sum = 0;
            for (int i_prev = 0; i_prev < layerSizes.get(l - 1); i_prev++) {
                double activation_prev = state.activations.get(l - 1).get(i_prev);
                double weight_connecting = weights.get(l - 1).get(i_prev).get(i_curr);
                sum += activation_prev * weight_connecting;
            }
            double bias = biases.get(l).get(i_curr);
            zValuesOfCurrentLayer.add(sum + bias);
        }
        return zValuesOfCurrentLayer;
    }

    private void updateParams(NNState state, List<Double> observedValues) {
        List<List<Double>> biasGradient = new ArrayList<>();
        List<List<List<Double>>> weightGradient = new ArrayList<>();

        // output layer with squared error
        biasGradient.add(new ArrayList<>());
        for (int i = 0; i < layerSizes.getLast(); i++) {
            double diff = state.activations.getLast().get(i) - observedValues.get(i);
            biasGradient.getLast().add(
                    2 * diff
                            * afd(
                                    state.zValues.getLast().get(i),
                            activationFunctions.getLast()
                    )
            );
        }

        // other layers
        for (int l = layerSizes.size() - 2; l >= 0; l--) {
            biasGradient.addFirst(new ArrayList<>());
            weightGradient.addFirst(new ArrayList<>());
            for (int i_curr = 0; i_curr < layerSizes.get(l); i_curr++) {
                double da_dz = afd(state.zValues.get(l).get(i_curr), activationFunctions.get(l));
                double dC_da = 0;
                weightGradient.getFirst().add(new ArrayList<>());
                for (int i_next = 0; i_next < layerSizes.get(l + 1); i_next++) {
                    double nextBG = biasGradient.get(1).get(i_next);
                    dC_da += weights.get(l).get(i_curr).get(i_next) * nextBG;
                    weightGradient.getFirst().get(i_curr).add(state.activations.get(l).get(i_curr) * nextBG);
                }
                biasGradient.getFirst().add(da_dz * dC_da);
            }
        }


        if (weights.size() != weightGradient.size()) throw new DimensionalityException();
        for (int i = 0; i < weights.size(); i++) {
            if (weights.get(i).size() != weightGradient.get(i).size()) throw new DimensionalityException();

            for (int j = 0; j < weights.get(i).size(); j++) {
                if (weights.get(i).get(j).size() != weightGradient.get(i).get(j).size()) throw new DimensionalityException();

                for (int k = 0; k < weights.get(i).get(j).size(); k++) {
                    weights.get(i).get(j).set(
                            k,
                            weights.get(i).get(j).get(k) - learningRate * weightGradient.get(i).get(j).get(k)
                    );
                }
            }
        }

        if (biases.size() != biasGradient.size()) throw new DimensionalityException();
        for (int i = 0; i < biases.size(); i++) {
            if (biases.get(i).size() != biasGradient.get(i).size()) throw new DimensionalityException();

            for (int j = 0; j < biases.get(i).size(); j++) {
                biases.get(i).set(
                        j,
                        biases.get(i).get(j) - learningRate * biasGradient.get(i).get(j)
                );
            }
        }

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

    private double afd(double z, ActivationFunction functionType) {
        return switch (functionType) {
            case Identity -> 1;
            case ReLU -> z > 0 ? 1 : 0;
            case Sigmoid -> {
                double sigmoidResult = 1 / (1 + Math.exp(-z));
                yield sigmoidResult * (1 - sigmoidResult);
            }
            case Tanh -> {
                double tanhResult = Math.tanh(z);
                yield 1 - tanhResult * tanhResult;
            }
            default -> throw new ActivationFunctionException("Use of not implemented activation function: " + functionType);
        };
    }

    private List<List<Double>> getRandomBiases() {
        List<List<Double>> biases_ret = new ArrayList<>();

        for (int l = 0; l < layerSizes.size(); l++) {
            biases_ret.add(new ArrayList<>());
            for (int i_curr = 0; i_curr < layerSizes.get(l); i_curr++) {
                biases_ret.get(l).add(randomDouble());
            }
        }
        return biases_ret;
    }

    private List<List<List<Double>>> getRandomWeights() {
        List<List<List<Double>>> weights_ret = new ArrayList<>();

        for (int l = 1; l < layerSizes.size(); l++) {
            weights_ret.add(new ArrayList<>());
            for (int i_prev = 0; i_prev < layerSizes.get(l - 1); i_prev++) {
                weights_ret.get(l - 1).add(new ArrayList<>());
                for (int i_curr = 0; i_curr < layerSizes.get(l); i_curr++) {
                    weights_ret.get(l - 1).get(i_prev).add(randomDouble());
                }
            }
        }
        return weights_ret;
    }

    @Override
    public void writeToFile(String fileName) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(System.getProperty("user.dir") + "/" + fileName))) {
            oos.writeObject(this);
            System.out.println(System.getProperty("user.dir") + "/" + fileName);
        }
    }

    public static Regressor createFromFile(String fileName) throws IOException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(System.getProperty("user.dir") + "/" + fileName))) {
            return (Regressor) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    private static double randomDouble() {
        return Math.random() * 2 - 1;
    }
}