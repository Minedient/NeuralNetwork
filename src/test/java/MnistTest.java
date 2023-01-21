import de.ratopi.mnist.read.io.MnistImageProvider;
import de.ratopi.mnist.read.io.MnistLabelProvider;
import func.MnistDataWorker;
import objects.Matrix;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

public class MnistTest {

    @Test
    void MnistDataReadTest() throws IOException {
        MnistImageProvider imageProvider = new MnistImageProvider(new File("D:\\Download\\train-images.idx3-ubyte"));
        MnistLabelProvider labelProvider = new MnistLabelProvider(new File("D:\\Download\\train-labels.idx1-ubyte"));
        MnistImageProvider testImageProvider = new MnistImageProvider(new File("D:\\Download\\t10k-images-idx3-ubyte.gz"));
        MnistLabelProvider testLabelProvider = new MnistLabelProvider(new File("D:\\Download\\t10k-labels-idx1-ubyte.gz"));
        for (int i = 0; i < 5002; i++) {
            imageProvider.selectNext();
            labelProvider.selectNext();
        }
        byte[] data = imageProvider.getCurrentData();
        for (int i = 0; i < imageProvider.getImageHeight(); i++) {
            for (int j = 0; j < imageProvider.getImageWidth(); j++) {
                System.out.print((data[j + i* imageProvider.getImageHeight()] == 0)?". ":"# ");
            }
            System.out.println();
        }
        System.out.println("It is a " + labelProvider.getCurrentValue());
    }

    @Test
    void MnistDataReadNormalizedTest() throws IOException {
        MnistImageProvider imageProvider = new MnistImageProvider(new File("D:\\Download\\train-images.idx3-ubyte"));
        imageProvider.selectNext();

        Matrix data = MnistDataWorker.toMatrix(imageProvider.getCurrentData());
        for (int i = 0; i < imageProvider.getImageHeight(); i++) {
            for (int j = 0; j < imageProvider.getImageWidth(); j++) {
                System.out.print(data.getDatum(j + i*imageProvider.getImageWidth(), 0) >=0.5?". ":"# ");
            }
            System.out.println();
        }
    }
}
