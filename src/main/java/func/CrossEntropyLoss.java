package func;

import obj.NeuralNetwork;
import objects.Matrix;

/**
 * Implemented following PyTorch's documentation
 */
public class CrossEntropyLoss implements LossFunction{

    private static double sum(double[] doubles){
        double d= 0.0;
        for (int i = 0; i < doubles.length; i++) {
            d += doubles[i];
        }
        return d;
    }
    @Override
    public double loss(Matrix values, Matrix expectedValues) {
        Matrix temp = Matrix.copyingMatrix(values);
        temp.forEach(v->Math.exp(v));
        return -values.getDatum(NeuralNetwork.findMaxIndexStatic(expectedValues), 0) + Math.log(sum(temp.getData()));
    }

    @Override
    public Matrix lossDerivative(Matrix values, Matrix expectedValues) {
        Matrix temp = Matrix.copyingMatrix(values);
        temp.forEach(v->Math.exp(v));
        temp = Matrix.scalarMultiplication(temp, 1 / sum(temp.getData()));
        double d = temp.getDatum(NeuralNetwork.findMaxIndexStatic(expectedValues), 0);
        temp.setDatum(NeuralNetwork.findMaxIndexStatic(expectedValues), 0, --d);
        return temp;
    }
}
