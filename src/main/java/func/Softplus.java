package func;

public class Softplus implements ActivationFunction{
    @Override
    public double activate(double values) {
        return Math.log(1 + Math.exp(values));
    }

    @Override
    public double derivative(double values) {
        return 1 / (1 + Math.exp(-values)); // aka sigmoid
    }
}
