package func;

/**
 * LeakyReLU, defined as max(x, 0.01x)
 */
public class LeakyReLU implements ActivationFunction{

    private static final double leakFactor = 0.01;

    @Override
    public double activate(double values) {
        return Math.max(values, values * leakFactor);
    }

    @Override
    public double derivative(double values) {
        return (values > 0)? 1 : leakFactor;
    }
}
