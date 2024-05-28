#include "Cmatrix.h"
#include "time.h"
#include <profileapi.h>
using namespace std;

int main() { 
    LARGE_INTEGER frequency, start, finish;
    ofstream fout ("benchmark.txt");
    for (int N = 1024; N <= 16384; N *= 2)
    {
        for (int n = 1; n <= 128; n *= 2)
        {
            vector<cl_float> a_data(n * N, 1);
            vector<cl_float> b_data(n * N, 2);
            Matrix a(a_data, n);
            a = T(a);
            Matrix b(b_data, n);
            QueryPerformanceFrequency(&frequency);
            QueryPerformanceCounter(&start);
            Matrix tmp = a * b;
            QueryPerformanceCounter(&finish);
            fout << (double)(finish.QuadPart - start.QuadPart) / frequency.QuadPart  * 1000 << ", ";
        }
        fout << '\n';
    }
    
   return 0;
}