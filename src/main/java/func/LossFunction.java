package func;

import objects.Matrix;

/**
 * Loss functions used to calculate the error
 */
public interface LossFunction {

    double loss(Matrix values, Matrix expectedValues);
    Matrix lossDerivative(Matrix values, Matrix expectedValues);

}
