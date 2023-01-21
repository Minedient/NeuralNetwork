package obj;

import exceptions.MatrixDimensionsNotMatchException;
import func.*;
import objects.Matrix;

import java.util.Random;

/**
 * This class represent a layer of neurons in a neural network.
 * <br>
 * In each layer (except for the input layer), contains an input weights Matrix, and a bias Matrix
 *
 * @author Minedient
 */
public class Layer {

    private final static Random random = new Random();
    private final int numOfInputs;
    private final int numOfOutputs;
    private final ActivationFunction activationFunction;
    private Matrix inputWeights;
    private Matrix bias;

    /**
     * Create a new {@code Layer} with given number of inputs and its blueprint
     *
     * @param numOfInputs The number of inputs
     * @param blueprint   The blueprint of the layer
     */
    public Layer(int numOfInputs, NeuralNetworkBlueprint blueprint) {
        this(numOfInputs, blueprint.layerSize(), blueprint.activationMode());
    }

    /**
     * Create a new {@code Layer} with given number of inputs, outputs and the activation function used.
     *
     * @param numOfInputs    The number of inputs
     * @param numOfOutputs   The number of outputs
     * @param activationMode The activation function used
     */
    public Layer(int numOfInputs, int numOfOutputs, ActivationMode activationMode) {
        this.numOfInputs = numOfInputs;
        this.numOfOutputs = numOfOutputs;
        switch (activationMode) {
            case LEAKY_RELU -> this.activationFunction = new LeakyReLU();
            case STEP -> this.activationFunction = new Step();
            case TANH -> this.activationFunction = new Tanh();
            case SOFT_PLUS -> this.activationFunction = new Softplus();
            default -> this.activationFunction = new Sigmoid();
        }
        this.inputWeights = Matrix.createNewFilledMatrix(numOfOutputs, numOfInputs, randomizeDoubleArray(numOfInputs * numOfOutputs));
        this.bias = Matrix.createNewFilledColumnVector(randomizeDoubleArray(numOfOutputs));
    }

    /**
     * Create a new randomized double array
     *
     * @param size The size of the double array
     * @return the array
     */
    private double[] randomizeDoubleArray(int size) {
        double[] a = new double[size];
        for (int i = 0; i < a.length; i++)
            a[i] = random.nextDouble() - 1;
        return a;
    }

    /**
     * Get the number of inputs of this layer
     *
     * @return The number of inputs
     */
    public int getNumOfInputs() {
        return this.numOfInputs;
    }

    /**
     * Get the matrix of input weights
     *
     * @return The matrix
     */
    public Matrix getInputWeights() {
        return inputWeights;
    }

    /**
     * Get the matrix of bias
     *
     * @return The matrix
     */
    public Matrix getBias() {
        return bias;
    }

    /**
     * Get the number of neuron in this layer
     *
     * @return The number of neurons
     */
    public int getNumOfNeurons() {
        return this.numOfOutputs;
    }

    /**
     * Print the weights out
     */
    public void printWeights() {
        System.out.println("Weights");
        System.out.println(inputWeights);
        System.out.println("Bias");
        System.out.println(bias);
    }

    /**
     * Perform a forward pass on the layer
     *
     * @param layerInputs The inputs of this layer
     * @return The output of this layer after forward pass
     */
    public Matrix forwardPass(Matrix layerInputs) {
        Matrix temp = null;
        try {
            temp = Matrix.addition(Matrix.multiplication(inputWeights, layerInputs), bias);
        } catch (MatrixDimensionsNotMatchException e) {
            throw new RuntimeException(e);
        }
        temp.forEach(activationFunction::activate);
        return temp;
    }

    /**
     * Perform a forward pass on the layer using multithreaded algorithms
     * <br>
     * Not necessarily faster than the single threaded version!
     * @see Layer#forwardPass(Matrix)
     *
     * @param layerInputs The inputs of this layer
     * @return The output of this layer after forward pass
     */
    public Matrix multithreadedForwardPass(Matrix layerInputs) {
        Matrix temp = null;
        try {
            temp = Matrix.addition(Matrix.multiThreadedMultiplication(inputWeights, layerInputs), bias);
        } catch (MatrixDimensionsNotMatchException e) {
            throw new RuntimeException(e);
        }
        temp.forEach(activationFunction::activate);
        return temp;
    }

    /**
     * Perform a forward pass on the layer using OpenCL
     * <br>
     * Not necessarily faster than the single threaded version!
     * @see Layer#forwardPass(Matrix)
     *
     * @param layerInputs The inputs of this layer
     * @return The output of this layer after forward pass
     */
    public Matrix gpuAcceleratedForwardPass(Matrix layerInputs, ActivationMode mode) {
        Matrix temp = null;
        temp = Matrix.clForwardPass(inputWeights, layerInputs, bias, mode.ordinal());
        //temp = Matrix.addition(Matrix.clMultiplication(inputWeights, layerInputs), bias);
        //temp.forEach(activationFunction::activate);
        return temp;
    }

    /**
     * Calculate the derivative for each number in the matrix
     * @param error The matrix (error matrix)
     * @return  The result
     */
    public Matrix derivativeForMatrix(Matrix error) {
        Matrix result = Matrix.copyingMatrix(error);
        result.forEach(activationFunction::derivative);
        return result;
    }

    /**
     * Update weights in this layer
     *
     * @param learningRate  Learning Rate of the neural network
     * @param weightsError  The error for weights
     * @param biasError     The error for bias
     */
    public void updateWeights(double learningRate, Matrix weightsError, Matrix biasError) {
        weightsError.forEach(value -> value * learningRate);
        biasError.forEach(value -> value * learningRate);
        try {
            inputWeights = Matrix.subtraction(inputWeights, weightsError);
            bias = Matrix.subtraction(bias, biasError);
        } catch (MatrixDimensionsNotMatchException e) {
            throw new RuntimeException(e);
        }
    }

}
