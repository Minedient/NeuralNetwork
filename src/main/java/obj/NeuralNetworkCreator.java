package obj;

import func.CrossEntropyLoss;
import func.LossFunction;
import func.MeanSquaredError;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkCreator{

    private int inputs;
    private List<NeuralNetworkBlueprint> layersBlueprint = new ArrayList<>();
    private boolean inputSet = false;
    private LossFunction lossFunction = new MeanSquaredError();
    private double learningRate = 0.001;

    private NeuralNetworkCreator(){};

    /**
     * Start create NeuralNetwork
     * @return New instance of {@code NeuralNetworkCreator}
     */
    public static NeuralNetworkCreator start(){
        return new NeuralNetworkCreator();
    }

    /**
     * Set the number of inputs of the new Neural Network
     * @param numOfInputs  Number of inputs
     * @return this {@code NeuralNetworkCreator}
     */
    public NeuralNetworkCreator setNumOfInputs(int numOfInputs){
        inputSet = true;
        inputs = numOfInputs;
        return this;
    }

    /**
     * Add a new layer in the new NeuralNetwork
     * @param layerSize The size of the layer
     * @param activationMode    The activation function used by the layer
     * @return this {@code NeuralNetworkCreator}
     */
    public NeuralNetworkCreator addLayer(int layerSize, ActivationMode activationMode){
        layersBlueprint.add(new NeuralNetworkBlueprint(layerSize, activationMode));
        return this;
    }

    /**
     * Set the learning rate of the new NeuralNetwork, by default it is 0.001
     * @param learningRate  The learing rate
     * @return this {@code NeuralNetworkCreator}
     */
    public NeuralNetworkCreator setLearningRate(double learningRate){
        this.learningRate = learningRate;
        return this;
    }

    /**
     * Use "Mean Squared Error" loss function to calculate the error in the new NeuralNetwork
     * @see MeanSquaredError
     * @return this {@code NeuralNetworkCreator}
     */
    public NeuralNetworkCreator lossFunctionMSE(){
        lossFunction = new MeanSquaredError();
        return this;
    }

    /**
     * Use "Cross Entropy Loss" loss function to calculate the error in the new NeuralNetwork
     * @return this {@code NeuralNetworkCreator}
     */
    public NeuralNetworkCreator lossFunctionCEL(){
        lossFunction = new CrossEntropyLoss();
        return this;
    }

    /**
     * Initialize the {@code NeuralNetwork} object
     * @return the new NeuralNetwork
     */
    public NeuralNetwork create(){
        if(!inputSet)
            throw new RuntimeException("Neural network does not have input or output, quitting!");
        List<Layer> layers = new ArrayList<>();
        for (int i = 0; i < layersBlueprint.size(); i++) {
            if(i == 0)
                layers.add(new Layer(inputs, layersBlueprint.get(i)));
            else
                layers.add(new Layer(layersBlueprint.get(i-1).layerSize(), layersBlueprint.get(i)));
        }
        NeuralNetwork result = new NeuralNetwork(layers, lossFunction);
        result.setLearningRate(learningRate);
        return result;
    }

}

record NeuralNetworkBlueprint(int layerSize, ActivationMode activationMode){}