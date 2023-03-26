package runtime;

import de.ratopi.mnist.read.io.MnistImageProvider;
import de.ratopi.mnist.read.io.MnistLabelProvider;
import exceptions.MatrixDimensionsNotMatchException;
import func.MnistDataWorker;
import obj.ActivationMode;
import obj.NeuralNetwork;
import obj.NeuralNetworkCreator;
import obj.TaggedDataSet;
import objects.Matrix;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Demo class, very ugly
 */
public class MnistTrainer {

    public static void main(String[] args){
        MnistImageProvider imageProvider;
        MnistLabelProvider labelProvider;
        MnistImageProvider testImageProvider;
        MnistLabelProvider testLabelProvider;
        List<TaggedDataSet> trainingData = new ArrayList<>();
        List<TaggedDataSet> testingData = new ArrayList<>();


        try {
            imageProvider = new MnistImageProvider(new File("path-to-file\\train-images.idx3-ubyte"));
            labelProvider = new MnistLabelProvider(new File("path-to-file\\train-labels.idx1-ubyte"));
            testImageProvider = new MnistImageProvider(new File("path-to-file\\t10k-images-idx3-ubyte.gz"));
            testLabelProvider = new MnistLabelProvider(new File("path-to-file\\t10k-labels-idx1-ubyte.gz"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        addNetworkDataToList(imageProvider, labelProvider, trainingData);
        addNetworkDataToList(testImageProvider, testLabelProvider, testingData);


        final int height = imageProvider.getImageHeight();
        final int width = imageProvider.getImageWidth();
        // Create a neural network with 2 hidden layer (16) (784->16->16->10)
        NeuralNetwork network = NeuralNetworkCreator.start().setNumOfInputs(height * width).addLayer(16, ActivationMode.LEAKY_RELU).addLayer(16, ActivationMode.LEAKY_RELU).addLayer(10, ActivationMode.LEAKY_RELU).setLearningRate(0.01).lossFunctionCEL().create();
        network.printNetworksLayer();

        long start = System.currentTimeMillis();

        // Time to test
        int correctGuesses = 0;
        double totalLoss = 0;
        for(TaggedDataSet tds:testingData){
            Matrix output = network.forwardPropagation(tds.inputs());
            if(network.findMaxIndex(output) == network.findMaxIndex(tds.expectedOutputs()))
                correctGuesses++;
            totalLoss += network.loss(output,tds.expectedOutputs());
        }
        System.out.println("Before training, the network guess " + correctGuesses + " out of " + testImageProvider.getNumberOfItems() +  " correctly");
        System.out.println("Achieved an accuracy of " + (correctGuesses * 1.0d / testImageProvider.getNumberOfItems()) * 100 + "%");
        System.out.println("Loss: " + totalLoss);

        // Training
        for (int i = 0; i < 100; i++) {
            long time = System.currentTimeMillis();
            System.out.print("Epoch " + (i+1) + "...");
            int times =0;
            /*
            CountDownLatch latch = new CountDownLatch(trainingData.size());
            for (int j = 0; j < trainingData.size(); j++) {
                TaggedDataSet tds = trainingData.get(j);
                monitor.giveRequest(()->{
                    try {
                        network.train(tds.inputs(),tds.expectedOutputs());
                    } catch (MatrixDimensionsNotMatchException e) {
                        throw new RuntimeException(e);
                    }
                    latch.countDown();
                });
            }

            try{
                latch.await();
            }catch(InterruptedException e){

            }
            */

            for(TaggedDataSet tds:trainingData) {
                try {
                    network.train(tds.inputs(), tds.expectedOutputs());
                } catch (MatrixDimensionsNotMatchException e) {
                    throw new RuntimeException(e);
                }
                times++;
            }
            System.out.print("Finished!");
            // Time to test
            correctGuesses = 0;
            totalLoss = 0;
            for(TaggedDataSet tds:testingData){
                Matrix output = network.forwardPropagation(tds.inputs());
                if(network.findMaxIndex(output) == network.findMaxIndex(tds.expectedOutputs()))
                    correctGuesses++;
                totalLoss += network.loss(output,tds.expectedOutputs());
            }
            System.out.print(" -> Loss: " + totalLoss);
            System.out.println(" -> Time taken: " + (System.currentTimeMillis() - time) / 1000.0 + "s");
        }

        // Time to test
        correctGuesses = 0;
        totalLoss = 0;
        for(TaggedDataSet tds:testingData){
            Matrix output = network.forwardPropagation(tds.inputs());
            if(network.findMaxIndex(output) == network.findMaxIndex(tds.expectedOutputs()))
                correctGuesses++;
            totalLoss += network.loss(output,tds.expectedOutputs());
        }
        System.out.println("After training, the network guess " + correctGuesses + " out of " + testImageProvider.getNumberOfItems() +  " correctly");
        System.out.println("Achieved an accuracy of " + (correctGuesses * 1.0d / testImageProvider.getNumberOfItems()) * 100 + "%");
        System.out.println("Loss: " + totalLoss);


        long end = System.currentTimeMillis();

        System.out.println("Time taken = " + (end-start) + "ms");

        Matrix.stopLibraryWorker();
    }

    private static void addNetworkDataToList(MnistImageProvider testImageProvider, MnistLabelProvider testLabelProvider, List<TaggedDataSet> testingData) {
        for (int i = 0; i < testImageProvider.getNumberOfItems(); i++) {
            try {
                testImageProvider.selectNext();
                testLabelProvider.selectNext();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            testingData.add(new TaggedDataSet(MnistDataWorker.toMatrix(testImageProvider.getCurrentData()), MnistDataWorker.toMatrix(testLabelProvider.getCurrentValue(),10)));
        }
    }
}
