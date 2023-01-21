package func;

import objects.Matrix;

import java.io.File;

public class MnistDataWorker {

    public static Matrix toMatrix(byte[] bytes){
        double[] matrixData = new double[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            //int gray = 255 - (bytes[i] & 255);
            //matrixData[i] = gray | gray << 8 | gray << 16;    //blow up

            //matrixData[i] = bytes[i];     // blow up

            //matrixData[i] = (bytes[i] & 255) / 255.0;
            matrixData[i] = (1 - (255 - (bytes[i] & 255)) / 255.0); // better than above
        }
        return Matrix.createNewFilledMatrix(bytes.length, 1, matrixData);
    }

    public static Matrix toMatrix(byte target, int range){
        Matrix result = Matrix.createNewEmptyColumnVector(range);
        result.setDatum(target, 0,1);
        return result;
    }

    public static Matrix loadDataFromCSV(File path){



        return Matrix.createNewEmptyMatrix(1,1);
    }


}
