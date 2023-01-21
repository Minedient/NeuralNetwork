import exceptions.MatrixDimensionsNotMatchException;
import objects.Matrix;
import org.junit.jupiter.api.Test;

public class LinearAlgebraPackageTest {



    @Test
    void cpuST10k() throws MatrixDimensionsNotMatchException {
        Matrix a = Matrix.createNewEmptyMatrix(16,784).fillRandomDoubles();
        Matrix b = Matrix.createNewEmptyMatrix(784,1).fillRandomDoubles();
        long start = System.currentTimeMillis();
        for (int i = 0; i < 10000; i++) {
            Matrix.multiplication(a,b);
        }
        System.out.println("Time taken (cpuST10k): " + (System.currentTimeMillis() - start)/1000.0 + "s");
    }

    @Test
    void cpuMT10k() throws MatrixDimensionsNotMatchException {
        Matrix a = Matrix.createNewEmptyMatrix(16,784).fillRandomDoubles();
        Matrix b = Matrix.createNewEmptyMatrix(784,1).fillRandomDoubles();
        long start = System.currentTimeMillis();
        for (int i = 0; i < 10000; i++) {
            Matrix.multiThreadedMultiplication(a,b);
        }
        System.out.println("Time taken (cpuMT10k): " + (System.currentTimeMillis() - start)/1000.0 + "s");
    }

    @Test
    void gpuST10k() throws MatrixDimensionsNotMatchException {
        Matrix a = Matrix.createNewEmptyMatrix(16,784).fillRandomDoubles();
        Matrix b = Matrix.createNewEmptyMatrix(784,1).fillRandomDoubles();

        Matrix.startOfLibrary();

        long start = System.currentTimeMillis();
        for (int i = 0; i < 10000; i++) {
            Matrix.clMultiplication(a,b);
        }
        System.out.println("Time taken (gpuST10k): " + (System.currentTimeMillis() - start)/1000.0 + "s");

        Matrix.endOfLibrary();
    }

}
