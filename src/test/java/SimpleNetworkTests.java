import exceptions.MatrixDimensionsNotMatchException;
import func.LeakyReLU;
import func.MeanSquaredError;
import obj.ActivationMode;
import obj.Layer;
import obj.NeuralNetwork;
import obj.NeuralNetworkCreator;
import objects.Matrix;
import org.junit.jupiter.api.Test;

public class SimpleNetworkTests {

    private double[] randomizeDoubleArray(int size){
        double[] a = new double[size];
        for (int i = 0; i < a.length; i++)
            a[i] = Math.random();
        return a;
    }

    @Test
    void weightsSum() throws MatrixDimensionsNotMatchException {
        Matrix inputs = Matrix.createNewFilledMatrix(2, 1, randomizeDoubleArray(2));
        Layer l = new Layer(2,3, ActivationMode.LEAKY_RELU);
        l.forwardPass(inputs);
    }

    @Test
    void networkCreationTest(){
        NeuralNetwork network = NeuralNetworkCreator.start().setNumOfInputs(2).addLayer(4, ActivationMode.LEAKY_RELU).addLayer(1, ActivationMode.LEAKY_RELU).create();
        System.out.println(network.forwardPropagation(Matrix.createNewFilledColumnVector(2,1)));
    }

    @Test
    void simpleNetworkForwardPassTest() throws MatrixDimensionsNotMatchException {
        NeuralNetwork network = NeuralNetworkCreator.start().setNumOfInputs(2).addLayer(1,ActivationMode.LEAKY_RELU).create();
        //network.printNetworksLayer();
        //network.printNetworksLayerDetails();
        Matrix inputs = Matrix.createNewFilledColumnVector(1,1);
        Matrix target = Matrix.createNewFilledColumnVector(0);
        System.out.println("Input is:");
        System.out.println(inputs);
        System.out.println("Output is:");
        Matrix output = network.forwardPropagation(Matrix.createNewFilledColumnVector(1,1));
        System.out.println(output);
        System.out.println("Loss is:");
        System.out.println(network.loss(output, target));

        // Time to train
        for (int i = 0; i < 100; i++) {
            network.train(inputs,target);
        }
        System.out.println("Input is:");
        System.out.println(inputs);
        System.out.println("Output is:");
        output = network.forwardPropagation(Matrix.createNewFilledColumnVector(1,1));
        System.out.println(output);
        System.out.println("Loss is:");
        System.out.println(network.loss(output, target));
    }

    @Test
    void LeakyReLUTest(){
        Matrix value = Matrix.createNewFilledColumnVector(-1,-2,3,4);
        LeakyReLU l = new LeakyReLU();
        value.forEach(l::activate);
        System.out.println(value);
        value.setColumn(0,new double[]{-1,-2,3,4});
        value.forEach(l::derivative);
        System.out.println(value);
    }

    @Test
    void MSETest(){
        Matrix value = Matrix.createNewFilledColumnVector(1,2,3,4);
        Matrix target = Matrix.createNewFilledColumnVector(0,0,0,0);
        MeanSquaredError mse = new MeanSquaredError();
        System.out.println(mse.loss(value,target));
        System.out.println(mse.lossDerivative(value,target));
    }

    @Test
    void simpleBigMatrixMultiplication() throws MatrixDimensionsNotMatchException {
        Matrix a = Matrix.createNewEmptyMatrix(2000,2000).fillRandomDoubles();
        Matrix b = Matrix.createNewEmptyMatrix(2000,2000).fillRandomDoubles();
        System.out.println(Matrix.clMultiplication(a,b));

        Matrix.endOfLibrary();
    }

}
