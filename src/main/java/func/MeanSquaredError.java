package func;

import exceptions.MatrixDimensionsNotMatchException;
import objects.Matrix;

/**
 * Find the error (or loss) of the network after one pass using Mean Squared Error approach
 */
public class MeanSquaredError implements LossFunction{
    @Override
    public double loss(Matrix values, Matrix expectedValues) {
        double result = 0.0;
        for (int i = 0; i < values.getColumnSize(); i++)
            result += Math.pow(values.getDatum(i, 0) - expectedValues.getDatum(i, 0), 2);
        return result / values.getColumnSize();
    }

    @Override
    public Matrix lossDerivative(Matrix values, Matrix expectedValues) {
        Matrix result = Matrix.copyingMatrix(values);
        try {
            result = Matrix.scalarMultiplication(Matrix.subtraction(result, expectedValues), 2.0 / result.getNumOfEntries());   // <= After three days, I finally catch this problem, I forget to reassign variable!
            // Matrix.scalarMultiplication(Matrix.subtraction(result, expectedValues), 2.0 / result.getNumOfEntries()); (Original code)
        } catch (MatrixDimensionsNotMatchException e) {
            throw new RuntimeException(e);
        }
        return result;
    }
}
