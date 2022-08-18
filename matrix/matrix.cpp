#include <vector>
#include <string>
#include <iostream>
#include <random>

using namespace std;

class Matrix
{
public:
    vector<vector<double>> mat;
    int row;
    int col;

    Matrix(int rows, int cols, double value)
    {
        mat.resize(rows);
        for (int i = 0; i < rows; i++)
        {
            mat[i].resize(cols, value);
        }
        row = rows;
        col = cols;
    }

    Matrix()
    {
        row = 0;
        col = 0;
    }

    void print()
    {
        printf("Rows: ");
        printf("%d", row);
        printf("\t Cols: ");
        printf("%d", col);
        printf("\n");
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                printf("%6.2f", mat[i][j]);
            }
            printf("\n");
        }
        fflush(stdout);
    }
    void printPixels()
    {
        printf("Rows: ");
        printf("%d", row);
        printf("\t Cols: ");
        printf("%d", col);
        printf("\n");
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                printf("%4.0f", mat[i][j] * 256);
            }
            printf("\n");
        }
    }

    void printInt()
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                printf("%4d", (int)mat[i][j]);
            }
            printf("\n");
        }
    }

    string toString()
    {
        string s = "";
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                s += to_string(mat[i][j]) + " ";
            }
            s += "\n";
        }
        return s;
    }

    Matrix copy(Matrix m)
    {
        return m;
    }

    double uniformDistribution(double low, double high, int key)
    {
        srand(key * time(0));
        double diff = high - low;
        int scale = 10000;
        int scaledDiff = (int)(diff * scale);
        return low + (1.0 * (rand() % scaledDiff) / scale);
    }

    void randomize(int n)
    {
        double min = -1.0 / sqrt(n);
        double max = 1.0 / sqrt(n);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                mat[i][j] = uniformDistribution(min, max, j + i);
            }
        }
    }

    void add(Matrix m)
    {
        if (row == m.row && col == m.col)
        {
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    mat[i][j] = mat[i][j] + m.mat[i][j];
                }
            }
        }
    }
};

// operations

Matrix flatten(Matrix m, bool toVert)
{

    if (!toVert)
    {
        Matrix out(1, m.row * m.col, -1);
        out.mat[0].clear();
        for (int i = 0; i < m.row; i++)
        {
            out.mat[0].insert(out.mat[0].end(), m.mat[i].begin(), m.mat[i].end());
        }
        return out;
    }
    else
    {
        Matrix out(m.row * m.col, 1, 0);
        out.mat.clear();
        for (int i = 0; i < m.col; i++)
        {
            for (int j = 0; j < m.row; j++)
            {
                out.mat.push_back(vector<double>{m.mat[j][i]});
            }
        }
        return out;
    }
}

Matrix dot(Matrix m1, Matrix m2)
{
    Matrix out(m1.row, m2.col, -1);
    if (m1.col == m2.row)
    {

        for (int i = 0; i < m1.row; i++)
        {
            for (int j = 0; j < m2.col; j++)
            {
                double sum = 0;
                for (int k = 0; k < m2.row; k++)
                {
                    sum += m1.mat[i][k] * m2.mat[k][j];
                }
                out.mat[i][j] = sum;
            }
        }
    }
    return out;
}

Matrix transpose(Matrix m)
{
    Matrix out(m.col, m.row, -1);
    for (int i = 0; i < m.row; i++)
    {
        for (int j = 0; j < m.col; j++)
        {
            out.mat[j][i] = m.mat[i][j];
        }
    }
    return out;
}

Matrix add(Matrix m1, Matrix m2)
{
    Matrix out(m1.row, m1.col, -1);
    if (m1.row == m2.row && m1.col == m2.col)
    {
        for (int i = 0; i < m1.row; i++)
        {
            for (int j = 0; j < m1.col; j++)
            {
                out.mat[i][j] = m1.mat[i][j] + m2.mat[i][j];
            }
        }
    }
    return out;
}

Matrix subtract(Matrix m1, Matrix m2)
{
    Matrix out(m1.row, m1.col, -1);
    if (m1.row == m2.row && m1.col == m2.col)
    {
        for (int i = 0; i < m1.row; i++)
        {
            for (int j = 0; j < m1.col; j++)
            {
                out.mat[i][j] = m1.mat[i][j] - m2.mat[i][j];
            }
        }
    }
    return out;
}

Matrix sigmoid(Matrix m)
{
    Matrix out(m.row, m.col, -1);
    for (int i = 0; i < m.row; i++)
    {
        for (int j = 0; j < m.col; j++)
        {
            out.mat[i][j] = (1.0 / (1.0 + (exp(-1 * m.mat[i][j]))));
        }
    }
    return out;
}

Matrix sigmoidPrime(Matrix m)
{
    Matrix out(m.row, m.col, -1);
    for (int i = 0; i < m.row; i++)
    {
        for (int j = 0; j < m.col; j++)
        {
            out.mat[i][j] = (1.0 / (1.0 + (exp(-1 * m.mat[i][j])))) * (1 - (1.0 / (1.0 + (exp(-1 * m.mat[i][j])))));
        }
    }
    return out;
}
double totalColExp(Matrix m, int column)
{
    double sum = 0.0;
    for (int i = 0; i < m.row; i++)
    {

        sum += exp(m.mat[i][column]);
    }
    return sum;
}

Matrix softmax(Matrix m)
{
    Matrix out(m.row, m.col, -1);
    double expsum = totalColExp(m, 0);
    for (int i = 0; i < m.row; i++)
    {
        for (int j = 0; j < m.col; j++)
        {
            out.mat[i][j] = exp(m.mat[i][j]) / expsum;
        }
    }
    return out;
}

Matrix multiply(Matrix m1, Matrix m2)
{
    Matrix out(m1.row, m1.col, -1);
    if (m1.row == m2.row && m1.col == m2.col)
    {
        for (int i = 0; i < m1.row; i++)
        {
            for (int j = 0; j < m1.col; j++)
            {
                out.mat[i][j] = m1.mat[i][j] * m2.mat[i][j];
            }
        }
    }
    return out;
}

Matrix scale(Matrix m, double scale)
{
    Matrix scalem(m.row, m.col, scale);
    return multiply(m, scalem);
}

int maxindex(Matrix m)
{
    double max = -10000;
    int index = -1;
    for (int i = 0; i < m.row; i++)
    {
        if (max < m.mat[i][0])
        {
            max = m.mat[i][0];
            index = i;
        }
    }
    return index;
}

// int main()
// {
//     Matrix m1(10, 1, 0);
//     Matrix m2(10, 1, 0);
//     m1.randomize(1);

//     m1.print();
//     printf("%d", maxindex(m1));
//     return 0;
// }