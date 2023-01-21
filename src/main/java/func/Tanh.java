package func;
public class Tanh implements ActivationFunction{
    @Override
    public double activate(double values) {
        return Math.tanh(values);
    }

    @Override
    public double derivative(double values) {
        return 1 - Math.pow(activate(values), 2);
    }
}
