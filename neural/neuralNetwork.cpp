#include "../util/images.cpp"

using namespace std;

struct neuralNetwork
{
    int input;
    int hidden;
    int output;
    double learningRate;
    Matrix W1;
    Matrix W2;
    Matrix B1;
    Matrix B2;

    neuralNetwork(int _input, int _hidden, int _output, double lr)
    {
        input = _input;
        hidden = _hidden;
        output = _output;
        learningRate = lr;
        new (&W1) Matrix(hidden, input, 0);
        new (&W2) Matrix(output, hidden, 0);
        new (&B1) Matrix(hidden, 1, 0);
        new (&B2) Matrix(output, 1, 0);
        W1.randomize(_hidden);
        W2.randomize(_output);
        B1.randomize(_hidden);
        B2.randomize(_output);
    }

    void train(Matrix A0, int y)
    {
        // make expected matrix
        Matrix expected(10, 1, 0);
        expected.mat[y][0] = 1;

        // forward propagation
        Matrix Z1 = add(dot(W1, A0), B1);
        Matrix A1 = sigmoid(Z1);
        Matrix Z2 = add(dot(W2, A1), B2);
        Matrix A2 = softmax(Z2);

        // back propagation
        Matrix dA2 = subtract(A2, expected);                                               // 1x10
        Matrix dZ2 = multiply(dA2, multiply(A2, subtract(Matrix(A2.row, A2.col, 1), A2))); // 1x10
        Matrix dW2 = dot(dZ2, transpose(A1));
        Matrix dB2 = dZ2.copy(dZ2);

        Matrix dZ1 = multiply(dot(transpose(W2), dZ2), sigmoidPrime(Z1));
        Matrix dW1 = dot(dZ1, transpose(A0));
        Matrix dB1 = dZ1.copy(dZ1);

        // update parameters
        W1 = subtract(W1, scale(dW1, learningRate));
        B1 = subtract(B1, scale(dB1, learningRate));

        W2 = subtract(W2, scale(dW2, learningRate));
        B2 = subtract(B2, scale(dB2, learningRate));
    }

    void trainAverage(Matrix A0, int y, Matrix *ddW1, Matrix *ddB1, Matrix *ddW2, Matrix *ddB2)
    {
        // make expected matrix
        Matrix expected(10, 1, 0);
        expected.mat[y][0] = 1;

        // forward propagation
        Matrix Z1 = add(dot(W1, A0), B1);
        Matrix A1 = sigmoid(Z1);
        Matrix Z2 = add(dot(W2, A1), B2);
        Matrix A2 = softmax(Z2);

        // back propagation
        Matrix dA2 = subtract(A2, expected);                                               // 1x10
        Matrix dZ2 = multiply(dA2, multiply(A2, subtract(Matrix(A2.row, A2.col, 1), A2))); // 1x10
        Matrix dW2 = dot(dZ2, transpose(A1));
        Matrix dB2 = dZ2.copy(dZ2);

        Matrix dZ1 = multiply(dot(transpose(W2), dZ2), sigmoidPrime(Z1));
        Matrix dW1 = dot(dZ1, transpose(A0));
        Matrix dB1 = dZ1.copy(dZ1);

        ddW1->add(dW1);
        ddB1->add(dB1);
        ddW2->add(dW2);
        ddB2->add(dB2);
    }

    void trainBatchAverage(vector<Img> imgs, int ammounttotrain, int averagesize)
    {
        for (int i = 0; i < ammounttotrain / averagesize; i++)
        {

            // initialize empty matricies
            Matrix *dW1 = new Matrix(W1.row, W1.col, 0);
            Matrix *dB1 = new Matrix(B1.row, B1.col, 0);
            Matrix *dW2 = new Matrix(W2.row, W2.col, 0);
            Matrix *dB2 = new Matrix(B2.row, B2.col, 0);

            for (int j = 0; j < averagesize; j++)
            {
                Img curimg = imgs[i * 10 + j];
                trainAverage(flatten(curimg.imgdata, true), curimg.label, dW1, dB1, dW2, dB2);
            }

            // update parameters
            W1 = subtract(W1, scale(*dW1, learningRate / averagesize));
            B1 = subtract(B1, scale(*dB1, learningRate / averagesize));

            W2 = subtract(W2, scale(*dW2, learningRate / averagesize));
            B2 = subtract(B2, scale(*dB2, learningRate / averagesize));
            // B2.print();

            if ((i * averagesize) % 100 == 0)
            {
                printf("Images Trained: ");
                printf("%d", 100 + i * averagesize);
                printf("/");
                printf("%d", ammounttotrain);
                printf("\n");
                fflush(stdout);
            }
        }
    }

    void trainBatch(vector<Img> imgs, int ammounttotrain)
    {
        for (int i = 0; i < ammounttotrain; i++)
        {
            Img curimg = imgs[i];
            train(flatten(curimg.imgdata, true), curimg.label);

            if (i % 100 == 0)
            {
                printf("Images Trained: ");
                printf("%d", 100 + i);
                printf("/");
                printf("%d", ammounttotrain);
                printf("\n");
                fflush(stdout);
            }
        }
    }

    Matrix predictImage(Img img)
    {
        Matrix A0 = flatten(img.imgdata, true);
        Matrix Z1 = add(dot(W1, A0), B1);
        Matrix A1 = sigmoid(Z1);
        Matrix Z2 = add(dot(W2, A1), B2);
        Matrix A2 = softmax(Z2);
        return A2;
    }

    double predictImages(vector<Img> imgs, int numberofimgs)
    {
        int numcorrect = 0;
        for (int i = 0; i < numberofimgs; i++)
        {
            Matrix prediction = predictImage(imgs[i]);
            if (maxindex(prediction) == imgs[i].label)
            {
                numcorrect++;
            }
            printf("Prediction: ");
            printf("%d", maxindex(prediction));
            printf("\n");
            imgs[i].print();
            fflush(stdout);
        }
        return 1.0 * numcorrect / numberofimgs;
    }

    void test(Matrix *m)
    {
        m->add(*m);
    }
};

int main()
{
    cout.flush();
    vector<Img> trainingImgs = CSVtoImgs("../images/mnist_train.csv", 59000);
    printf("Images loaded.\n");
    neuralNetwork network(784, 20, 10, 0.08);
    printf("network created.\n");
    network.trainBatch(trainingImgs, 59000);
    // network.trainBatchAverage(trainingImgs, 50000, 100);
    printf("Training complete.\n");

    vector<Img> testingImgs = CSVtoImgs("../images/mnist_test.csv", 1000);
    double accuracy = network.predictImages(testingImgs, 1000);
    printf("Accuracy: ");
    printf("%f", accuracy);

    return 0;
}