package libs.JML.util;

import java.util.ArrayList;
import java.util.List;

public class NNState {
    public final List<List<Double>> activations;
    public final List<List<Double>> zValues;

    public NNState() {
        activations = new ArrayList<>();
        zValues = new ArrayList<>();
    }

    @Override
    public String toString() {
        return "zValues: " + zValues + ",\nactivations: " + activations;
    }
}
