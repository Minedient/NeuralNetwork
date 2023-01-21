import func.ActivationFunction;
import func.LeakyReLU;
import func.Sigmoid;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ActivationFunctionTests {

    ActivationFunction sigmoid = new Sigmoid();
    ActivationFunction leaky = new LeakyReLU();

    @Test
    void sigmoidTest(){
        double x = 1.0;
        assertEquals(1/ (1 + Math.exp(-x)), sigmoid.activate(x));
        assertEquals(sigmoid.activate(x) * (1 - sigmoid.activate(x)), sigmoid.derivative(x));
        System.out.println(sigmoid.activate(x));
        System.out.println(sigmoid.derivative(x));
    }

    @Test
    void leakyTest(){
        double x = 2.0, y = -2.0;
        assertEquals(2, leaky.activate(x));
        assertEquals(-0.02, leaky.activate(y));
        assertEquals(1, leaky.derivative(x));
        assertEquals(0.01, leaky.derivative(y));
    }
}
