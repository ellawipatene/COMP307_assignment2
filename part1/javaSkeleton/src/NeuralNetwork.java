import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;

    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
    }


    //Calculate neuron activation for an input
    public double sigmoid(double input) {
        double output = Double.NaN; //TODO!
        return output;
    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;
            output_layer_outputs[i] = output;
        }
        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int desired_outputs) {

        double[] output_layer_betas = new double[num_outputs];
        // TODO! Calculate output layer betas.
        System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        double[] hidden_layer_betas = new double[num_hidden];
        // TODO! Calculate hidden layer betas.
        System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        // TODO! Calculate output layer weight changes.

        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        // TODO! Calculate hidden layer weight changes.

        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights};
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) {
        // TODO! Update the weights
        System.out.println("Placeholder");
    }

    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);
                int predicted_class = -1; // TODO!
                predictions[i] = predicted_class;

                //We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1]);
            }

            // Print new weights
            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            // TODO: Print accuracy achieved over this epoch
            double acc = Double.NaN;
            System.out.println("acc = " + acc);
        }
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            int predicted_class = -1;  // TODO !Should be 0, 1, or 2.
            predictions[i] = predicted_class;
        }
        return predictions;
    }

}
