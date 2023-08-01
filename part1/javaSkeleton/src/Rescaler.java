public class Rescaler {
    private final double[] mins;
    private final double[] maxes;

    public Rescaler(double[][] data) {
        this.mins = new double[data[0].length];
        this.maxes = new double[data[0].length];

        for (int j = 0; j < data[0].length; j++) {
            mins[j] = Double.MAX_VALUE;
            maxes[j] = -Double.MAX_VALUE;
        }

        for (double[] instance : data) {
            for (int j = 0; j < instance.length; j++) {
                mins[j] = Math.min(mins[j], instance[j]);
                maxes[j] = Math.max(maxes[j], instance[j]);
            }
        }

    }

    public void rescaleData(double[][] data) {
        for (double[] instance : data) {
            for (int j = 0; j < instance.length; j++) {
                instance[j] = (instance[j] - mins[j]) / (maxes[j] - mins[j]);
            }
        }
    }

}
