#include <fstream>
#include "..\matrix\matrix.cpp"
#include <string>
#include <string.h>
#include <iostream>
#include <charconv>

#define MAXCHAR 10000

using namespace std;

struct Img
{
    Matrix imgdata;
    int label;

    Img(int _label, Matrix _imgdata)
    {
        imgdata = _imgdata;
        label = _label;
    }

    void print()
    {
        printf("label: ");
        printf("%d", label);
        printf("\n");
        imgdata.printPixels();
    }
};

vector<Img> CSVtoImgs(string file_name, int number_of_images)
{
    vector<Img> imgs;

    ifstream file;
    file.open(file_name);
    char row[MAXCHAR];
    int images_loaded = 0;

    // skip first line
    file.ignore(MAXCHAR, '\n');

    // read in file lines
    while (file.good() && images_loaded < number_of_images)
    {
        int _label = -1;
        Matrix m(28, 28, 0);

        file.getline(row, MAXCHAR, '\n');
        char *token = strtok(row, ",");
        vector<string> array;

        _label = stoi(token);
        token = strtok(NULL, ",");

        while (token != NULL)
        {
            array.push_back(token);
            token = strtok(NULL, ",");
        }
        int row = 0;
        int col = 0;
        for (int i = 0; i < array.size(); i++)
        {
            m.mat[row][col] = stod(array[i]) / 256;
            if (col < 27)
            {
                col++;
            }
            else
            {
                row++;
                col = 0;
            }
        }
        imgs.push_back(Img(_label, m));
        images_loaded++;
        if (images_loaded % 100 == 0)
        {
            printf("Images Loaded: ");
            printf("%d", images_loaded);
            printf("/");
            printf("%d", number_of_images);
            printf("\n");
            fflush(stdout);
        }
    }
    file.close();
    return imgs;
}

// int main()
// {
//     vector<Img> imgs = CSVtoImgs("../images/mnist_train.csv", 1);
//     for (int i = 0; i < imgs.size(); i++)
//     {
//         imgs[i].print();
//     }
//     printf("%f", imgs[0].imgdata.mat[8][8]);
//     return 0;
// }