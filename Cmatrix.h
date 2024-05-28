#pragma once
#include "Matrix.h"

class Cmatrix
{
public:
    Matrix real, imag;
    Cmatrix() {}
    Cmatrix(const Cmatrix& other) : real(other.real), imag(other.imag) {}
    Cmatrix(Matrix real, Matrix imag) : real(real), imag(real) {}
    Cmatrix(Matrix real) : real(real), imag(Matrix(vector<cl_float>(real.h * real.w, 0), real.h)) {}
    Cmatrix(int h, int w) : real(Matrix(h, w)), imag (Matrix(h, w)) {}
};

ostream& operator<< (ostream &os, const Cmatrix &a);
Cmatrix operator* (const Cmatrix& a, const Cmatrix& b);
Cmatrix operator+ (const Cmatrix& a, const Cmatrix& b);
Cmatrix operator% (const Cmatrix& a, const Cmatrix& b);
Cmatrix compExp(const Matrix& a);//Применяет комплексную экспоненту (exp(1i * a)) к матрице
Cmatrix T(const Cmatrix& a);
