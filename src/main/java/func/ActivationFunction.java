package func;

/**
 * The activation function used by a {@code Layer}
 */
public interface ActivationFunction {
    double activate(double values);

    double derivative(double values);

}
