import de.ratopi.mnist.read.io.MnistImageProvider;
import de.ratopi.mnist.read.io.MnistLabelProvider;
import exceptions.MatrixDimensionsNotMatchException;
import func.MnistDataWorker;
import obj.ActivationMode;
import obj.NeuralNetwork;
import obj.NeuralNetworkCreator;
import objects.Matrix;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

public class ComplexNetworkTests {
    @Test
    void MnistDatasetTest() throws IOException {
        MnistImageProvider imageProvider = new MnistImageProvider(new File("D:\\Download\\train-images.idx3-ubyte"));
        MnistLabelProvider labelProvider = new MnistLabelProvider(new File("D:\\Download\\train-labels.idx1-ubyte"));
        MnistImageProvider testImageProvider = new MnistImageProvider(new File("D:\\Download\\t10k-images-idx3-ubyte.gz"));
        MnistLabelProvider testLabelProvider = new MnistLabelProvider(new File("D:\\Download\\t10k-labels-idx1-ubyte.gz"));
        final int height = imageProvider.getImageHeight();
        final int width = imageProvider.getImageWidth();
        // Create a neural network with 2 hidden layer (16) (784->16->16->10)
        NeuralNetwork network = NeuralNetworkCreator.start().setNumOfInputs(height * width).addLayer(16, ActivationMode.LEAKY_RELU).addLayer(16, ActivationMode.LEAKY_RELU).addLayer(10, ActivationMode.LEAKY_RELU).setLearningRate(0.01).create();
        network.printNetworksLayer();

        // Before training
        int correctGuesses = 0;
        double totalLoss = 0;
        for (int i = 0; i < testImageProvider.getNumberOfItems(); i++) {
            testLabelProvider.selectNext();
            testImageProvider.selectNext();
            Matrix input = MnistDataWorker.toMatrix(testImageProvider.getCurrentData());
            Matrix output = network.forwardPropagation(input);
            Matrix target = MnistDataWorker.toMatrix(testLabelProvider.getCurrentValue(), 10);
            if(network.findMaxIndex(output) == (int) testLabelProvider.getCurrentValue())
                correctGuesses++;
            totalLoss += network.loss(output,target);
        }
        System.out.println("Before training, the network guess " + correctGuesses + " out of " + testImageProvider.getNumberOfItems() +  " correctly");
        System.out.println("Achieved an accuracy of " + (correctGuesses * 1.0d / testImageProvider.getNumberOfItems()) * 100 + "%");
        System.out.println("Loss: " + totalLoss);

        // Training time!

        for (int i = 0; i < imageProvider.getNumberOfItems(); i++) {
            imageProvider.selectNext();
            labelProvider.selectNext();
            Matrix input = MnistDataWorker.toMatrix(imageProvider.getCurrentData());
            Matrix target = MnistDataWorker.toMatrix(labelProvider.getCurrentValue(), 10);
            Matrix output = network.forwardPropagation(input);
            if(i % 1000 == 0 || i < 10){
                System.out.println(i + "     " + network.loss(output, target) + "     "+  Matrix.transpose(output));
            }
            try {
                network.train(input, target);
            } catch (MatrixDimensionsNotMatchException e) {
                throw new RuntimeException(e);
            }
        }
        totalLoss = 0.0;
        correctGuesses = 0;

        testImageProvider = new MnistImageProvider(new File("D:\\Download\\t10k-images-idx3-ubyte.gz"));
        testLabelProvider = new MnistLabelProvider(new File("D:\\Download\\t10k-labels-idx1-ubyte.gz"));

        for (int i = 0; i < testImageProvider.getNumberOfItems(); i++) {
            testLabelProvider.selectNext();
            testImageProvider.selectNext();
            Matrix input = MnistDataWorker.toMatrix(testImageProvider.getCurrentData());
            Matrix output = network.forwardPropagation(input);
            Matrix target = MnistDataWorker.toMatrix(testLabelProvider.getCurrentValue(), 10);
            if(network.findMaxIndex(output) == (int) testLabelProvider.getCurrentValue())
                correctGuesses++;
            totalLoss += network.loss(output,target);
        }
        System.out.println("After training, the network guess " + correctGuesses + " out of " + testImageProvider.getNumberOfItems() +  " correctly");
        System.out.println("Achieved an accuracy of " + (correctGuesses * 1.0d / testImageProvider.getNumberOfItems()) * 100 + "%");
        System.out.println("Loss: " + totalLoss);
    }

    @Test
    void quadrantsTest(){
        // Try to match coordinate that is either larger than or equal to (1,1) or smaller than or equal to (-1,-1) in a [-2,2]×[-2,2] square
        NeuralNetwork network = NeuralNetworkCreator.start().setNumOfInputs(2).addLayer(10, ActivationMode.LEAKY_RELU).addLayer(2, ActivationMode.LEAKY_RELU).create();
        Matrix[] testingData = new Matrix[1000];
        Matrix[] testingDataLabel = new Matrix[1000];
        Matrix[] trainingData = new Matrix[100000];
        Matrix[] trainingDataLabel = new Matrix[100000];
        for (int i = 0; i < testingData.length; i++) {
            double[] d = new double[2];
            double x = 4*Math.random()-2;
            double y = 4*Math.random()-2;
            testingData[i] = Matrix.createNewFilledColumnVector(x,y);
            if((x >= 1 && y >= 1) || (x <= -1 && y <= -1))
                d[0] = 1;
            else d[1] = 1;
            testingDataLabel[i] = Matrix.createNewFilledColumnVector(d);
        }
        for (int i = 0; i < trainingData.length; i++) {
            double[] d = new double[2];
            double x = 4*Math.random()-2;
            double y = 4*Math.random()-2;
            trainingData[i] = Matrix.createNewFilledColumnVector(x,y);
            if((x >= 1 && y >= 1) || (x <= -1 && y <= -1))
                d[0] = 1;
            else d[1] = 1;
            trainingDataLabel[i] = Matrix.createNewFilledColumnVector(d);
        }

        // Test
        int correctGuesses = 0;
        double totalLoss = 0;
        for (int i = 0; i < testingData.length; i++) {
            Matrix input = testingData[i];
            Matrix output = network.forwardPropagation(input);
            Matrix target = testingDataLabel[i];
            if(network.findMaxIndex(output) == network.findMaxIndex(target))
                correctGuesses++;
            totalLoss += network.loss(output,target);
        }
        System.out.println("Before training, the network guess " + correctGuesses + " out of " + 1000 +  " correctly");
        System.out.println("Achieved an accuracy of " + (correctGuesses * 1.0d / 1000) * 100 + "%");
        System.out.println("Loss: " + totalLoss);

        // Training time!

        for(int k=0;k<10;k++) {
            for (int i = 0; i < trainingData.length; i++) {
                try {
                    network.train(trainingData[i], trainingDataLabel[i]);
                } catch (MatrixDimensionsNotMatchException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        correctGuesses = 0;
        totalLoss = 0;
        for (int i = 0; i < testingData.length; i++) {
            Matrix input = testingData[i];
            Matrix output = network.forwardPropagation(input);
            Matrix target = testingDataLabel[i];
            if(network.findMaxIndex(output) == network.findMaxIndex(target))
                correctGuesses++;
            totalLoss += network.loss(output,target);
        }
        System.out.println("After training, the network guess " + correctGuesses + " out of " + 1000 +  " correctly");
        System.out.println("Achieved an accuracy of " + (correctGuesses * 1.0d / 1000) * 100+ "%");
        System.out.println("Loss: " + totalLoss);
    }



}
