package obj;

import exceptions.MatrixDimensionsNotMatchException;
import func.LossFunction;
import objects.Matrix;

import java.util.*;

public class NeuralNetwork {
    private final Map<String, Object> settingsMap = new HashMap<>();
    private List<Layer> layers;
    private LossFunction lossFunction;

    NeuralNetwork(List<Layer> layers, LossFunction lossFunction) {
        this.layers = layers;
        this.lossFunction = lossFunction;
    }

    public static int findMaxIndexStatic(Matrix matrix) {
        int maxIndex = 0;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < matrix.getColumnSize(); i++) {
            if (max < matrix.getDatum(i, 0)) {
                maxIndex = i;
                max = matrix.getDatum(i, 0);
            }
        }
        return maxIndex;
    }

    public void setLearningRate(double learningRate) {
        settingsMap.put("learningRate", Double.valueOf(learningRate));
    }

    public Matrix forwardPropagation(Matrix inputs) {
        Matrix passingMatrix = Matrix.copyingMatrix(inputs);
        for (Layer layer : layers) {
            passingMatrix = layer.forwardPass(passingMatrix);
        }
        return passingMatrix;
    }

    public Matrix clForwardPropagation(Matrix inputs) {
        Matrix passingMatrix = Matrix.copyingMatrix(inputs);
        for (Layer layer : layers) {
            passingMatrix = layer.gpuAcceleratedForwardPass(passingMatrix, ActivationMode.LEAKY_RELU);
        }
        return passingMatrix;
    }

    public double loss(Matrix outputs, Matrix expected) {
        // The outputs Matrix and the expected Matrix should be a column vector
        return lossFunction.loss(outputs, expected);
    }

    public void train(Matrix inputs, Matrix expected) throws MatrixDimensionsNotMatchException {
        // Store intermediate results generated during training for easy access.
        LinkedList<Matrix> intermediateResults = new LinkedList<>();
        intermediateResults.add(inputs);

        // ForwardPropagation
        for (Layer layer : layers)
            intermediateResults.add(layer.forwardPass(intermediateResults.get(intermediateResults.size() - 1)));
        //intermediateResults.add(layer.gpuAcceleratedForwardPass(intermediateResults.get(intermediateResults.size()-1), ActivationMode.LEAKY_RELU));

        // Find the loss of the network
        Matrix lossMatrix = lossFunction.lossDerivative(intermediateResults.pollLast(), expected);
        // BackwardPropagation
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Matrix currentLayerOutput = intermediateResults.get(i);

            // Find the gradients

            // Error from Sum and ActivationFunctions

            Matrix y = Matrix.addition(Matrix.multiplication(currentLayer.getInputWeights(), currentLayerOutput), currentLayer.getBias());
            // Matrix y = currentLayer.forwardPass(currentLayerOutput);             <= After three days, I finally catch this problem
            Matrix errorFromBias = currentLayer.derivativeForMatrix(y).linearMultiplication(lossMatrix);

            // Generate the error information for next layer
            lossMatrix = Matrix.multiplication(Matrix.transpose(currentLayer.getInputWeights()), errorFromBias);

            // Error from the weights of this layer
            Matrix errorFromWeights = Matrix.multiplication(errorFromBias, Matrix.transpose(currentLayerOutput));

            // Update the weights
            currentLayer.updateWeights((Double) settingsMap.get("learningRate"), errorFromWeights, errorFromBias);
        }
    }

    public void multithreadedTrain(Matrix inputs, Matrix expected) throws MatrixDimensionsNotMatchException {
        // Store intermediate results generated during training for easy access.
        LinkedList<Matrix> intermediateResults = new LinkedList<>();
        intermediateResults.add(inputs);

        // ForwardPropagation
        for (Layer layer : layers)
            intermediateResults.add(layer.multithreadedForwardPass(intermediateResults.get(intermediateResults.size() - 1)));

        // Find the loss of the network
        Matrix lossMatrix = lossFunction.lossDerivative(intermediateResults.pollLast(), expected);
        // BackwardPropagation
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Matrix currentLayerOutput = intermediateResults.get(i);

            // Find the gradients

            // Error from Sum and ActivationFunctions

            Matrix y = Matrix.addition(Matrix.multiThreadedMultiplication(currentLayer.getInputWeights(), currentLayerOutput), currentLayer.getBias());
            // Matrix y = currentLayer.forwardPass(currentLayerOutput);             <= After three days, I finally catch this problem
            Matrix errorFromBias = currentLayer.derivativeForMatrix(y).linearMultiplication(lossMatrix);

            // Generate the error information for next layer
            lossMatrix = Matrix.multiThreadedMultiplication(Matrix.transpose(currentLayer.getInputWeights()), errorFromBias);

            // Error from the weights of this layer
            Matrix errorFromWeights = Matrix.multiThreadedMultiplication(errorFromBias, Matrix.transpose(currentLayerOutput));

            // Update the weights
            currentLayer.updateWeights((Double) settingsMap.get("learningRate"), errorFromWeights, errorFromBias);
        }
    }

    public void gpuAcceleratedTrain(Matrix inputs, Matrix expected) throws MatrixDimensionsNotMatchException {
        // Store intermediate results generated during training for easy access.
        LinkedList<Matrix> intermediateResults = new LinkedList<>();
        intermediateResults.add(inputs);

        // ForwardPropagation
        for (Layer layer : layers)
            intermediateResults.add(layer.forwardPass(intermediateResults.get(intermediateResults.size() - 1)));

        // Find the loss of the network
        Matrix lossMatrix = lossFunction.lossDerivative(intermediateResults.pollLast(), expected);
        // BackwardPropagation
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Matrix currentLayerOutput = intermediateResults.get(i);

            // Find the gradients

            // Error from Sum and ActivationFunctions

            Matrix y = Matrix.addition(Matrix.clMultiplication(currentLayer.getInputWeights(), currentLayerOutput), currentLayer.getBias());
            // Matrix y = currentLayer.forwardPass(currentLayerOutput);             <= After three days, I finally catch this problem
            Matrix errorFromBias = currentLayer.derivativeForMatrix(y).linearMultiplication(lossMatrix);

            // Generate the error information for next layer
            lossMatrix = Matrix.clMultiplication(Matrix.transpose(currentLayer.getInputWeights()), errorFromBias);

            // Error from the weights of this layer
            Matrix errorFromWeights = Matrix.clMultiplication(errorFromBias, Matrix.transpose(currentLayerOutput));

            // Update the weights
            currentLayer.updateWeights((Double) settingsMap.get("learningRate"), errorFromWeights, errorFromBias);
        }
    }
    private void multiThreadedBatchTraining(TaggedDataSet set, Matrix[][] errorFromBatchesWeights, Matrix[][] errorFromBatchesBias, int finalJ) throws MatrixDimensionsNotMatchException {
        LinkedList<Matrix> intermediateResults = new LinkedList<>();
        intermediateResults.add(set.inputs());

        // ForwardPropagation
        for (Layer layer : layers)
            intermediateResults.add(layer.forwardPass(intermediateResults.get(intermediateResults.size() - 1)));

        Matrix lossMatrix = lossFunction.lossDerivative(intermediateResults.pollLast(), set.expectedOutputs());

        for (int k = layers.size() - 1; k >= 0; k--) {
            Layer currentLayer = layers.get(k);
            Matrix currentLayerOutput = intermediateResults.get(k);

            Matrix y = Matrix.addition(Matrix.multiplication(currentLayer.getInputWeights(), currentLayerOutput), currentLayer.getBias());

            errorFromBatchesBias[k][finalJ] = currentLayer.derivativeForMatrix(y).linearMultiplication(lossMatrix);

            lossMatrix = Matrix.multiplication(Matrix.transpose(currentLayer.getInputWeights()), errorFromBatchesBias[k][finalJ]);

            errorFromBatchesWeights[k][finalJ] = Matrix.multiplication(errorFromBatchesBias[k][finalJ], Matrix.transpose(currentLayerOutput));
        }
    }

    public void printNetworksLayer() {
        System.out.print(layers.get(0).getNumOfInputs());
        for (Layer layer : layers)
            System.out.print("->" + layer.getNumOfNeurons());
        System.out.println();
    }

    public void printNetworksLayerDetails() {
        for (Layer layer : layers) {
            layer.printWeights();
        }
    }

    public int findMaxIndex(Matrix matrix) {
        int maxIndex = 0;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < matrix.getColumnSize(); i++) {
            if (max < matrix.getDatum(i, 0)) {
                maxIndex = i;
                max = matrix.getDatum(i, 0);
            }
        }
        return maxIndex;
    }
}