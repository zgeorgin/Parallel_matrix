#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <string>
#include <fstream>
#include <time.h>
#include <functional>
#include <cmath>
#include <profileapi.h>

using namespace std;



class Matrix
{
public:
    vector<cl_float> data;
    int h, w;
    Matrix(vector<cl_float> data, int h) : data(data), h(h), w(data.size() / h) {}
    Matrix(const Matrix& other) : data(other.data), h(other.h), w(other.w) {}
    Matrix(int h, int w) : data(vector<cl_float> (h * w, 0)), h(h), w(w) {}
    Matrix() {}
    void InsertMat(int start_h, int start_w, Matrix mat);
};

ostream& operator<< (ostream &os, const Matrix &a);
Matrix operator* (const Matrix& a, const Matrix& b);
Matrix operator* (cl_float b, const Matrix& a);
Matrix operator* (const Matrix& a, cl_float b);
Matrix operator/ (cl_float b, const Matrix& a);
Matrix operator/ (const Matrix& a, cl_float b);
Matrix operator+ (const Matrix& a, const Matrix& b);
Matrix operator- (const Matrix& a, const Matrix& b);
Matrix operator+ (cl_float b, const Matrix& a);
Matrix operator+ (const Matrix& a, cl_float b);
Matrix operator- (cl_float b, const Matrix& a);
Matrix operator- (const Matrix& a, cl_float b);
Matrix operator% (const Matrix& a, const Matrix& b); // Поэлементное умножение

Matrix range(double begin, double end, double step);

double norm(const Matrix& a, std::string type);
Matrix abs(const Matrix& a);
double deg2rad(double deg);
Matrix deg2rad(const Matrix& deg);
Matrix applyFuncToMatrix(std::function<cl_float(cl_float)> func, const Matrix& a);
Matrix T(const Matrix& a);
Matrix E(int size);

#endif