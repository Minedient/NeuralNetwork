package func;

public class Sigmoid implements ActivationFunction{
    @Override
    public double activate(double values) {
        return 1 / (1 + Math.exp(-values));
    }

    @Override
    public double derivative(double values) {
        return activate(values) * (1 - activate(values));
    }
}
