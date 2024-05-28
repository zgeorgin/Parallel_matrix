__kernel void MatrixMult(__global const float* a, __global const float* b, __global float* c, int Aw, int Ah, int Bw, int Bh)
{
    // Получаем индекс в массиве
    int iGID = get_global_id(0);

    int y = iGID / Bw;
    int x = iGID - y * Bw;

    // Проверяем, не вылезли ли за пределы массива
    if (iGID >= Bw * Ah)
    {   
        return; 
    }
    float sum = 0;

    for (int i = 0; i < Aw; i++)
    {
        sum += a[i + Aw * y] * b[i * Bw + x];
    }
    
    c[iGID] = sum;
}

__kernel void MatrixInsert(__global const float* a, __global const float* b, __global float* c, int Aw, int Ah, int Bw, int Bh, int start_w, int start_h)
{
    // Получаем индекс в массиве
    int iGID = get_global_id(0);

    int y = iGID / Aw;
    int x = iGID - y * Aw;

    // Проверяем, не вылезли ли за пределы массива
    if (iGID >= Aw * Ah)
    {   
        return; 
    }
    
    if (x >= start_w && x < start_w + Bw && y >= start_h && y < start_h + Bh)
        c[iGID] = b[(y - start_h) * Bw + (x - start_w)];
    else
        c[iGID] = a[y * Aw + x];
}

__kernel void MatrixTranspose(__global const float* a, __global float* c, int Aw, int Ah)
{
    // Получаем индекс в массиве
    int iGID = get_global_id(0);

    int y = iGID / Aw;
    int x = iGID - y * Aw;

    // Проверяем, не вылезли ли за пределы массива
    if (iGID >= Aw * Ah)
    {   
        return; 
    }
    
    c[x * Ah + y] = a[iGID];
}

__kernel void MatrixElMult(__global const float* a, __global const float* b, __global float* c, int Aw, int Ah, int Bw, int Bh)
{
    int iGID = get_global_id(0);

    if (iGID >= Aw * Ah)
    {
        return;
    }

    c[iGID] = a[iGID] * b[iGID];
}

__kernel void MatrixSum(__global const float* a, __global const float* b, __global float* c, int Aw, int Ah, int Bw, int Bh)
{
    int iGID = get_global_id(0);

    if (iGID >= Aw * Ah)
    {
        return;
    }

    c[iGID] = a[iGID] + b[iGID];
}

__kernel void MatrixNumMult(__global const float* a, __global float* c, int Aw, int Ah, float num)
{
    int iGID = get_global_id(0);

    if (iGID >= Aw * Ah)
    {
        return;
    }

    c[iGID] = a[iGID] * num;
}

__kernel void MatrixNumSum(__global const float* a, __global float* c, int Aw, int Ah, int num)
{
    int iGID = get_global_id(0);

    if (iGID >= Aw * Ah)
    {
        return;
    }

    c[iGID] = a[iGID] + num;
}

__kernel void MatrixInvert(__global const float* a, __global float* c, int Aw, int Ah, int num)
{
    int iGID = get_global_id(0);

    if (iGID >= Aw * Ah)
    {
        return;
    }

    c[iGID] = 1 / a[iGID];
}

__kernel void MatrixAbs(__global const float* a, __global float* c, int Aw, int Ah)
{
    int iGID = get_global_id(0);

    if (iGID >= Aw * Ah)
    {
        return;
    }

    if (a[iGID] < 0) 
        c[iGID] = -a[iGID]; 
    else c[iGID] = a[iGID];
}