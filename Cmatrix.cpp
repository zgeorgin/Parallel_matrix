#include "Cmatrix.h"

Cmatrix operator* (const Cmatrix& a, const Cmatrix& b)
{
    Cmatrix res;
    res.real = (a.real * b.real) - (a.imag * b.imag);
    res.imag = (a.real * b.imag) + (a.imag * b.real);

    return res;
}

Cmatrix operator+ (const Cmatrix& a, const Cmatrix& b)
{
    Cmatrix res;
    res.real = a.real + b.real;
    res.imag = a.imag + b.imag;

    return res;
}

Cmatrix operator% (const Cmatrix& a, const Cmatrix& b) 
{
    Cmatrix res;
    res.real = a.real % b.real - a.imag % b.imag;
    res.imag = a.real % b.imag + a.imag % b.real;
    
    return res;
}

Cmatrix compExp(const Matrix& a) 
{
    Cmatrix res;
    function<cl_float(cl_float)> Msin = [] (cl_float x) { return (cl_float) sin(x); };
    function<cl_float(cl_float)> Mcos = [] (cl_float x) { return (cl_float) cos(x); };;
    res.imag = applyFuncToMatrix(Msin, a);
    res.real = applyFuncToMatrix(Mcos, a);

    return res;
}

ostream& operator<< (ostream &os, const Cmatrix &a)
{
    for (int i = 0; i < a.real.h; i++)
    {
        for (int j = 0; j < a.real.w; j++)
        {
            os << a.real.data[i * a.real.w + j] << "+i*" << a.imag.data[i * a.real.w + j] << " ";
        }
        os << '\n';
    }
    return os;
}

Cmatrix T(const Cmatrix& a)
{
    Cmatrix res;
    res.real = T(a.real);
    res.imag = T(a.imag);
    return res;
}