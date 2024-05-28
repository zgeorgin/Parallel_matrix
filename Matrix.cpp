#include "Matrix.h"

double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)(end - start); // convert nanoseconds to seconds on return
}

void invoke_kernel_Mult(cl_kernel kernel, cl_command_queue queue, size_t localworksize, size_t globalworksize, cl_mem buff, cl_mem matA_buff, cl_mem matB_buff, cl_float* result, cl_float* matA, cl_float* matB, cl_int Aw, cl_int Ah, cl_int Bw, cl_int Bh) {  //Пример функции, подающей аргументы в kernel и запускающей его. Можно переделать под свою задачу
    cl_int err = 0;
    cl_event execution;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matA_buff); 
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matB_buff);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&buff);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&Aw);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&Ah);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&Bw);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&Bh);


       LARGE_INTEGER frequency, start, finish;

    err |= clEnqueueWriteBuffer(queue, matA_buff, CL_FALSE, 0, sizeof(cl_float) * Aw * Ah, matA, 0, NULL, NULL); 
    err |= clEnqueueWriteBuffer(queue, matB_buff, CL_FALSE, 0, sizeof(cl_float) * Bw * Bh, matB, 0, NULL, NULL);

    // запускаем одномерную задачу
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalworksize, &localworksize, 0, NULL, &execution);
    
    clFlush(queue);
    clWaitForEvents(1, &execution);
    
    //cout << "real time: " << executionTime(execution) << '\n';

    // читаем результат
    
    err |= clEnqueueReadBuffer(queue, buff, CL_TRUE, 0, sizeof(cl_float) * Bw * Ah, result, 0, NULL, NULL);
    QueryPerformanceCounter(&finish);
    // ждём завершения всех операций
        
    //cout << (double)(finish.QuadPart - start.QuadPart) / frequency.QuadPart  * 1000 << '\n';

    clFinish(queue);
    

}

void invoke_kernel_Transpose(cl_kernel kernel, cl_command_queue queue, size_t localworksize, size_t globalworksize, cl_mem buff, cl_mem matA_buff, cl_float* result, cl_float* matA, cl_int Aw, cl_int Ah) {  //Пример функции, подающей аргументы в kernel и запускающей его. Можно переделать под свою задачу
    cl_int err = 0;

    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matA_buff); 
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buff);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&Aw);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&Ah);

    err |= clEnqueueWriteBuffer(queue, matA_buff, CL_FALSE, 0, sizeof(cl_float) * Aw * Ah, matA, 0, NULL, NULL); 

    // запускаем одномерную задачу
    err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalworksize, &localworksize, 0, NULL, NULL);

    // читаем результат
    err |= clEnqueueReadBuffer(queue, buff, CL_TRUE, 0, sizeof(cl_float) * Aw * Ah, result, 0, NULL, NULL);

    // ждём завершения всех операций
    clFinish(queue);
}

void invoke_kernel_num(cl_kernel kernel, cl_command_queue queue, size_t localworksize, size_t globalworksize, cl_mem buff, cl_mem matA_buff, cl_float* result, cl_float* matA, cl_int Aw, cl_int Ah, cl_float num) {  //Пример функции, подающей аргументы в kernel и запускающей его. Можно переделать под свою задачу
    cl_int err = 0;

    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matA_buff); 
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buff);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&Aw);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&Ah);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&num);

    err |= clEnqueueWriteBuffer(queue, matA_buff, CL_FALSE, 0, sizeof(cl_float) * Aw * Ah, matA, 0, NULL, NULL); 

    // запускаем одномерную задачу
    err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalworksize, &localworksize, 0, NULL, NULL);

    // читаем результат
    err |= clEnqueueReadBuffer(queue, buff, CL_TRUE, 0, sizeof(cl_float) * Aw * Ah, result, 0, NULL, NULL);

    // ждём завершения всех операций
    clFinish(queue);
}

void invoke_kernel_Insert(cl_kernel kernel, cl_command_queue queue, size_t localworksize, size_t globalworksize, cl_mem buff, cl_mem matA_buff, cl_mem matB_buff, cl_float* result, cl_float* matA, cl_float* matB, cl_int Aw, cl_int Ah, cl_int Bw, cl_int Bh, cl_int start_h, cl_int start_w) {  //Пример функции, подающей аргументы в kernel и запускающей его. Можно переделать под свою задачу
    cl_int err = 0;

    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matA_buff); 
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matB_buff);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&buff);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&Aw);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&Ah);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&Bw);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&Bh);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&start_w);
    err |= clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&start_h);


    err |= clEnqueueWriteBuffer(queue, matA_buff, CL_FALSE, 0, sizeof(cl_float) * Aw * Ah, matA, 0, NULL, NULL); 
    err |= clEnqueueWriteBuffer(queue, matB_buff, CL_FALSE, 0, sizeof(cl_float) * Bw * Bh, matB, 0, NULL, NULL);

    // запускаем одномерную задачу

    err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalworksize, &localworksize, 0, NULL, NULL);

    // читаем результат
    err |= clEnqueueReadBuffer(queue, buff, CL_TRUE, 0, sizeof(cl_float) * Aw * Ah, result, 0, NULL, NULL);

    // ждём завершения всех операций
    clFinish(queue);
}

cl_device_id create_device() { //Определить устройство для исполнения программы
    cl_platform_id platform;
    cl_device_id dev;
    cl_int err = 0;
    char name[100];
    size_t name_size;
    err |= clGetPlatformIDs(1, &platform, NULL); // определяем платформу
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL); //Пытаемся выбрать GPU
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(char) * 100,(void *)&name, &name_size);
    //cout << name << '\n';
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL); //Если не получается, выбираем CPU
    }
    if (err) throw;
    return dev;
}

std::string get_program_text(std::string filepath) { //Получить весь текст программы из файла
    std::ifstream t(filepath);
    return std::string((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());
}

cl_program build_program(cl_context ctx, cl_device_id dev, std::string filepath) { //Собираем программу из kernel по пути filepath
    int err;

    std::string src = get_program_text(filepath);
    const char* src_text = src.data();
    size_t src_length = src.size();
    cl_program program = clCreateProgramWithSource(ctx, 1, &src_text, &src_length, &err);
    err |= clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    char* log;
    // тут желательно получить лог компиляции через clGetProgramBuildInfo
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 1000, log, NULL);
    
    if (err) throw log;

    return program;
}

std::vector<cl_float> MatrixMult(std::vector<cl_float> matA, std::vector<cl_float> matB, int Ah, int Bh, int Aw, int Bw, int localworksize, int globalworksize)
{
    

   cl_int err;
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   cl_program program = build_program(context, device, "MatrixMult.cl");
   cl_kernel kernel = clCreateKernel(program, "MatrixMult", &err);
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

   cl_mem matA_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); //Пример создания массива для kernel
   cl_mem matB_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Bh * Bw, NULL, &err); 
   cl_mem result_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Ah * Bw, NULL, &err); 

   
    std::vector<cl_float> result(Ah * Bw, 1.0);//(Ah * Bw, 2.0);

    invoke_kernel_Mult(kernel, queue, localworksize, globalworksize, result_buff, matA_buff, matB_buff, result.data(), matA.data(), matB.data(), Aw, Ah, Bw, Bh);

    // Освобождаем ресурсы
    clReleaseKernel(kernel);
    clReleaseMemObject(result_buff);
    clReleaseMemObject(matA_buff);
    clReleaseMemObject(matB_buff);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context); 

    return result;
}

std::vector<cl_float> MatrixInsert(std::vector<cl_float> matA, std::vector<cl_float> matB, int Ah, int Bh, int Aw, int Bw, int start_h, int start_w, int localworksize, int globalworksize)
{
   cl_int err;
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   cl_program program = build_program(context, device, "MatrixMult.cl");
   cl_kernel kernel = clCreateKernel(program, "MatrixInsert", &err);
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

   cl_mem matA_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); 
   cl_mem matB_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Bh * Bw, NULL, &err); 
   cl_mem result_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); 

   
    std::vector<cl_float> result(Ah * Aw, 1.0);//(Ah * Bw, 2.0);
   
    invoke_kernel_Insert(kernel, queue, localworksize, globalworksize, result_buff, matA_buff, matB_buff, result.data(), matA.data(), matB.data(), Aw, Ah, Bw, Bh, start_h, start_w);

    // Освобождаем ресурсы
    clReleaseKernel(kernel);
    clReleaseMemObject(result_buff);
    clReleaseMemObject(matA_buff);
    clReleaseMemObject(matB_buff);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return result;
}

std::vector<cl_float> MatrixElMult(std::vector<cl_float> matA, std::vector<cl_float> matB, int Ah, int Bh, int Aw, int Bw, int localworksize, int globalworksize)
{
   cl_int err;
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   cl_program program = build_program(context, device, "MatrixMult.cl");
   cl_kernel kernel = clCreateKernel(program, "MatrixElMult", &err);
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

   cl_mem matA_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); //Пример создания массива для kernel
   cl_mem matB_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Bh * Bw, NULL, &err); 
   cl_mem result_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Ah * Bw, NULL, &err); 

   
    std::vector<cl_float> result(Ah * Bw, 1.0);//(Ah * Bw, 2.0);
   
    invoke_kernel_Mult(kernel, queue, localworksize, globalworksize, result_buff, matA_buff, matB_buff, result.data(), matA.data(), matB.data(), Aw, Ah, Bw, Bh);

    // Освобождаем ресурсы
    clReleaseKernel(kernel);
    clReleaseMemObject(result_buff);
    clReleaseMemObject(matA_buff);
    clReleaseMemObject(matB_buff);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return result;
}

std::vector<cl_float> MatrixSum(std::vector<cl_float> matA, std::vector<cl_float> matB, int Ah, int Bh, int Aw, int Bw, int localworksize, int globalworksize)
{
   cl_int err;
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   cl_program program = build_program(context, device, "MatrixMult.cl");
   cl_kernel kernel = clCreateKernel(program, "MatrixSum", &err);
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

   cl_mem matA_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); //Пример создания массива для kernel
   cl_mem matB_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Bh * Bw, NULL, &err); 
   cl_mem result_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Ah * Bw, NULL, &err); 

   
    std::vector<cl_float> result(Ah * Bw, 1.0);//(Ah * Bw, 2.0);
   
    invoke_kernel_Mult(kernel, queue, localworksize, globalworksize, result_buff, matA_buff, matB_buff, result.data(), matA.data(), matB.data(), Aw, Ah, Bw, Bh);

    // Освобождаем ресурсы
    clReleaseKernel(kernel);
    clReleaseMemObject(result_buff);
    clReleaseMemObject(matA_buff);
    clReleaseMemObject(matB_buff);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return result;
}

std::vector<cl_float> MatrixTranspose(std::vector<cl_float> matA, int Ah, int Aw, int localworksize, int globalworksize)
{
   cl_int err;
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   cl_program program = build_program(context, device, "MatrixMult.cl");
   cl_kernel kernel = clCreateKernel(program, "MatrixTranspose", &err);
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

   cl_mem matA_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); //Пример создания массива для kernel
   cl_mem result_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); 

   
    std::vector<cl_float> result(Ah * Aw, 1.0);//(Ah * Bw, 2.0);
   
    invoke_kernel_Transpose(kernel, queue, localworksize, globalworksize, result_buff, matA_buff, result.data(), matA.data(), Aw, Ah);

    // Освобождаем ресурсы
    clReleaseKernel(kernel);
    clReleaseMemObject(result_buff);
    clReleaseMemObject(matA_buff);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return result;
}

std::vector<cl_float> MatrixNumMult(std::vector<cl_float> matA, cl_float num, int Ah, int Aw, int localworksize, int globalworksize)
{
   cl_int err;
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   cl_program program = build_program(context, device, "MatrixMult.cl");
   cl_kernel kernel = clCreateKernel(program, "MatrixNumMult", &err);
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

   cl_mem matA_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); //Пример создания массива для kernel
   cl_mem result_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); 

   
    std::vector<cl_float> result(Ah * Aw, 1.0);//(Ah * Bw, 2.0);
   
    invoke_kernel_num(kernel, queue, localworksize, globalworksize, result_buff, matA_buff, result.data(), matA.data(), Aw, Ah, num);

    // Освобождаем ресурсы
    clReleaseKernel(kernel);
    clReleaseMemObject(result_buff);
    clReleaseMemObject(matA_buff);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return result;
}

std::vector<cl_float> MatrixNumSum(std::vector<cl_float> matA, int num, int Ah, int Aw, int localworksize, int globalworksize)
{
   cl_int err;
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   cl_program program = build_program(context, device, "MatrixMult.cl");
   cl_kernel kernel = clCreateKernel(program, "MatrixNumSum", &err);
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

   cl_mem matA_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); //Пример создания массива для kernel
   cl_mem result_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); 

   
    std::vector<cl_float> result(Ah * Aw, 1.0);//(Ah * Bw, 2.0);
   
    invoke_kernel_num(kernel, queue, localworksize, globalworksize, result_buff, matA_buff, result.data(), matA.data(), Aw, Ah, num);

    // Освобождаем ресурсы
    clReleaseKernel(kernel);
    clReleaseMemObject(result_buff);
    clReleaseMemObject(matA_buff);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return result;
}

std::vector<cl_float> MatrixInvert(std::vector<cl_float> matA, int Ah, int Aw, int localworksize, int globalworksize)
{
   cl_int err;
   cl_device_id device = create_device();
   cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

   cl_program program = build_program(context, device, "MatrixMult.cl");
   cl_kernel kernel = clCreateKernel(program, "MatrixInvert", &err);
   cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

   cl_mem matA_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); //Пример создания массива для kernel
   cl_mem result_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); 

   
    std::vector<cl_float> result(Ah * Aw, 1.0);//(Ah * Bw, 2.0);
   
    invoke_kernel_num(kernel, queue, localworksize, globalworksize, result_buff, matA_buff, result.data(), matA.data(), Aw, Ah, 1);

    // Освобождаем ресурсы
    clReleaseKernel(kernel);
    clReleaseMemObject(result_buff);
    clReleaseMemObject(matA_buff);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return result;
}

std::vector<cl_float> MatrixAbs(std::vector<cl_float> matA, int Ah, int Aw, int localworksize, int globalworksize)
{
    cl_int err;
    cl_device_id device = create_device();
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    cl_program program = build_program(context, device, "MatrixMult.cl");

    cl_kernel kernel = clCreateKernel(program, "MatrixAbs", &err);
    
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    cl_mem matA_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err);
    cl_mem result_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Ah * Aw, NULL, &err); 

    
        std::vector<cl_float> result(Ah * Aw, 1.0);
    
        invoke_kernel_Transpose(kernel, queue, localworksize, globalworksize, result_buff, matA_buff, result.data(), matA.data(), Aw, Ah);

        // Освобождаем ресурсы
        clReleaseKernel(kernel);
        clReleaseMemObject(result_buff);
        clReleaseMemObject(matA_buff);
        clReleaseCommandQueue(queue);
        clReleaseProgram(program);
        clReleaseContext(context);

        return result;
}


Matrix operator* (const Matrix& a, const Matrix& b)
{
    std::vector<cl_float> data = 
        MatrixMult(a.data, b.data, a.h, b.h, a.w, b.w, 100, a.h * b.w  + (100 - a.h*b.w % 100));
    
    return Matrix(data, a.h);    
}

ostream& operator << (ostream &os, const Matrix &a)
{
    for (int i = 0; i < a.h; i++)
    {
        for (int j = 0; j < a.w; j++)
        {
            os << a.data[i*a.w + j] << ' ';
        }
        os << '\n';
    }
    return os;
}

Matrix range(double begin, double end, double step)
{
    vector<cl_float> res((int)((end - begin)/step));
    
    for (int i = 0; i < res.size(); i++) res[i] = step*i;
    return Matrix(res, 1);
}

double deg2rad(double deg)  {return deg * CL_M_PI / 180.0;}
Matrix deg2rad(const Matrix& deg)
{
    Matrix res(deg);
    for (int i = 0; i < res.data.size(); i++) res.data[i] = deg2rad(res.data[i]);
    return res;
}

Matrix operator+ (const Matrix& a, const Matrix& b)
{
    Matrix res(a);
    for (int i = 0; i < res.data.size(); i++) res.data[i] = a.data[i] + b.data[i];
    return res;
}

Matrix T(const Matrix& a)
{
    std::vector<cl_float> data = 
        MatrixTranspose(a.data, a.h, a.w, 100, a.h * a.w  + (100 - a.h*a.w % 100));
    return Matrix(data, a.w);
}

Matrix operator% (const Matrix& a, const Matrix& b)
{
    std::vector<cl_float> data =
        MatrixElMult(a.data, b.data, a.h, b.h, a.w, b.w, 100, a.h * a.w  + (100 - a.h*a.w % 100));
    return Matrix(data, a.h);
}

Matrix operator* (cl_float b, const Matrix& a)
{
    std::vector<cl_float> data =
        MatrixNumMult(a.data, b, a.h, a.w, 100, a.h * a.w  + (100 - a.h*a.w % 100));
    return Matrix(data, a.h);
}

Matrix operator* (const Matrix& a, cl_float b) { return b * a; }

Matrix operator- (const Matrix& a, const Matrix& b) { return a + (-1) * b; }

Matrix applyFuncToMatrix(std::function<cl_float(cl_float)> func, const Matrix& a)
{
    Matrix res = a;
    for (int i = 0; i < res.data.size(); i++) res.data[i] = func(res.data[i]);

    return res;
}

Matrix operator+ (cl_float b, const Matrix& a)
{
    std::vector<cl_float> data =
        MatrixNumSum(a.data, b, a.h, a.w, 100, a.h * a.w  + (100 - a.h*a.w % 100));
    return Matrix(data, a.h);
}

Matrix operator+ (const Matrix& a, cl_float b) { return b + a; }

Matrix operator- (cl_float b, const Matrix& a) { return b + (-1) * a; }

Matrix operator- (const Matrix& a, cl_float b) { return a + (-1) * b; }

Matrix operator/ (cl_float b, const Matrix& a)
{
    std::vector<cl_float> data =
        MatrixInvert(a.data, a.h, a.w, 100, a.h * a.w  + (100 - a.h*a.w % 100));
    return Matrix(data, a.h)  * b;
}

Matrix operator/ (const Matrix& a, cl_float b) { return a * (1.0 / b);}

void Matrix::InsertMat(int start_h, int start_w, Matrix mat)
{
    vector<cl_float> new_data = 
        MatrixInsert(data, mat.data, h, mat.h, w, mat.w, start_h, start_w, 100, h * w  + (100 - h*w % 100));
        
    data = new_data;
}

Matrix abs(const Matrix& a)
{
    std::vector<cl_float> result = MatrixAbs(a.data, a.h, a.w, 100, a.h * a.w  + (100 - a.h*a.w % 100));
    return Matrix(result, a.h);
}

double norm(const Matrix& a, std::string type)
{
    if (type == "inf")
    {
        std::vector<cl_float> ones(a.w, 1);
        Matrix b = Matrix(ones, 1);
        Matrix goal = b * abs(a);
        return *std::max_element(goal.data.begin(), goal.data.end());
    }

    if (type == "1")
    {
        std::vector<cl_float> ones(a.h, 1);
        Matrix b = Matrix(ones, a.h);
        Matrix goal = abs(a) * b;
        return *std::max_element(goal.data.begin(), goal.data.end());
    }

    if (type == "sum")
    {
        std::vector<cl_float> ones(a.h, 1);
        Matrix b = Matrix(ones, 1);
        Matrix goal = b * abs(a) * T(b);
        return goal.data[0];
    }

    return 0;
}

Matrix E(int size)
{
    std::vector<cl_float> data(size * size, 0);

    for (int i = 0; i < size; i++)
        data[i * size + i] = 1;
    
    return Matrix(data, size);
}