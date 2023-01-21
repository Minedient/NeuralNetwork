package func;

public class Step implements ActivationFunction{
    @Override
    public double activate(double values) {
        return (values >= 0)?1:0;
    }

    @Override
    public double derivative(double values) {
        return 0;
    }
}
