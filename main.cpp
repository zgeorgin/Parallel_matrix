#include "Cmatrix.h"
#include "time.h"
#include <random>
#include <profileapi.h>
using namespace std;

Matrix NewtonInvert(const Matrix& A, int stepsCount, double threshold = 0)
{
    Matrix cur = T(A) / (norm(A, "1") * norm (A, "inf"));
    for (int i = 0; i < stepsCount; i++)
    {
        cur = 2 * cur - cur * A * cur;
        if (norm(A * cur - E(A.h), "sum") < threshold) 
            break;
    }
    return cur;
}

int main() 
{
    int size = 5;
    std::vector<cl_float> A_data(size * size);
    srand(time(NULL));
    for (int i = 0; i < A_data.size(); i++)
    {
        A_data[i] = rand();
    }
    Matrix A(A_data, size);
    std::ofstream fout("test.txt");
    fout << A << '\n';
    Matrix A_inv = NewtonInvert(A, 150000, 1);
    fout << A_inv << '\n';
    fout << A * A_inv << '\n';
}

