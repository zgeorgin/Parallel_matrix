#include "CMatrix.h"
void testMatrixMult() //Тест матричного умножения
{
    vector<cl_float> a_data = {1, 0, 0, 1};
    vector<cl_float> b_data = {1, 0, 0, 1};

    Matrix a (a_data, 2);
    Matrix b (b_data, 2);
    ofstream fout("test.txt");
    fout << a << '\n';
    fout << b << '\n';
    fout << a * b;
}

void testMatrixElMult() //Тест поэлементнового умножения
{
    vector<cl_float> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    vector<cl_float> b_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    Matrix a (a_data, 3);
    Matrix b (b_data, 3);
    ofstream fout("test.txt");
    fout << a << '\n';
    fout << b << '\n';
    fout << a % b;
}

void testMatrixSum() //Тест сложения
{
    vector<cl_float> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    vector<cl_float> b_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    Matrix a (a_data, 3);
    Matrix b (b_data, 3);
    ofstream fout("test.txt");
    fout << a << '\n';
    fout << b << '\n';
    fout << a + b;
}

void testMatrixTranspose() //Тест трансонирования
{
    vector<cl_float> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix a(a_data, 3);
    ofstream fout("test.txt");
    fout << a << '\n';
    fout << T(a);
}

void testMatrixNumMult() //Тест умножения на число
{
    vector<cl_float> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix a(a_data, 3);
    ofstream fout("test.txt");
    fout << a << '\n';
    fout << a * 5;
}

void testMatrixNumSum() //Тест сложения с числом
{
    vector<cl_float> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix a(a_data, 3);
    ofstream fout("test.txt");
    fout << a << '\n';
    fout << a - 5;
}

void testMatrixDivision() //Тест деления на число (или деления числа на матрицу)
{
    vector<cl_float> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix a(a_data, 3);
    ofstream fout("test.txt");
    fout << a << '\n';
    fout << a /5;
}

void testComplex() // Тест комплексных матриц
{
    vector<cl_float> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix a(a_data, 3);
    ofstream fout("test.txt");
    fout << compExp(a);
}

void testInsert()
{
    vector<cl_float> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Matrix a(a_data, 3);
    vector<cl_float> b_data = {13, 14, 15, 16};
    Matrix b(b_data, 2);

    ofstream fout("test.txt");
    fout << a << '\n' << b << '\n';
    a.InsertMat(1, 1, b);
    fout << a;
}

void testAbs()
{
    vector<cl_float> a_data = {-1, 2, -3, 4, 5, -6, -7, -8, 9};
    Matrix a(a_data, 3);

    ofstream fout("test.txt");
    fout << abs(a) << '\n';
}

void testNorm()
{
    vector<cl_float> a_data = {-50, 2, -3, 30, 5, -6, -7, -8, 9};
    Matrix a(a_data, 3);

    ofstream fout("test.txt");
    fout << norm(a, "inf") << '\n' << norm(a, "1") << '\n' << norm(a, "sum") << '\n';
}

void testE()
{
    ofstream fout("test.txt");
    fout << E(5) <<'\n';
}

int main()
{
    //Раскомментировать нужный тест
    //testMatrixMult();
    //testMatrixTranspose();
    //testMatrixElMult();
    //testMatrixSum();
    //testMatrixNumMult();
    //testComplex();
    //testMatrixNumSum();
    //testMatrixDivision();
    //testInsert();
    //testAbs();
    //testNorm();
    //testE();
}