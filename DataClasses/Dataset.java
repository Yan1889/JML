package libs.JML.DataClasses;

import libs.JML.Exceptions.DimensionalityException;

import java.util.ArrayList;
import java.util.List;

public class Dataset {
    private final List<DataPoint> dataPoints;

    private final int dimensionX;
    private final int dimensionY;

    public Dataset(int dimensionX, int dimensionY) {
        this.dimensionX = dimensionX;
        this.dimensionY = dimensionY;
        dataPoints = new ArrayList<>();
    }

    public void putDataPoint(DataPoint newDataPoint) {
        if (dimensionX != newDataPoint.x().size() || dimensionY != newDataPoint.y().size()) throw new DimensionalityException("Cannot add Point with other dimensionality to dataset");
        dataPoints.add(newDataPoint);
    }

    public List<DataPoint> getDataPoints() {
        return dataPoints;
    }
    public int getDimensionX() {
        return dimensionX;
    }
    public int getDimensionY() {
        return dimensionY;
    }
}