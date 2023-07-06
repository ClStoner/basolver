#include "include/mymatrix.h"

namespace BaSolver
{

void tranlate(const double *R, const double *t, const double *p, double *res)
{
    for(int i = 0; i < 3; i ++)
    {
        res[i] = 0;
        for(int j = 0; j < 3; j ++)
        {
            res[i] += R[i * 3 + j] * p[j];
        }
        res[i] += t[i];
    }
}
void invTranlate(const double *R, const double *t, double *p, double *res)
{
    double tmp[3];
    for(int i = 0; i < 3; i ++) tmp[i] = p[i] - t[i];
    for(int i = 0; i < 3; i ++)
    {
        res[i] = 0;
        for(int j = 0; j < 3; j ++)
        {
            res[i] += R[j * 3 + i] * tmp[j];
        }
    }
}
void jacTjac(const double *A, const double *B, const int n, const int m, const int p, double *res)
{
    for(int i = 0; i < n; i ++)
    {
        for(int k = 0; k < p; k ++)
        {
            for(int j = 0; j < m; j ++)
            {
                res[i * m + j] += A[k * n + i] * B[k * m + j];
            }
        }
    }
}
void jacTjac(const double *A, const double *B, const int n, const int m, const int p, const int idx_i, const int idx_j, const int sz, double *res)
{
    for(int i = 0; i < n; i ++)
    {
        int x = (i + idx_i) * sz + idx_j; 
        for(int k = 0; k < p; k ++)
        {
            int km = k * m;
            int kn = k * n;
            for(int j = 0; j < m; j ++)
            {
                res[x + j] += A[kn + i] * B[km + j];
            }
        }
    }
}
void jacTres(const double *A, const double *R, const int n, const int p, double *res)
{
    for(int i = 0; i < n; i ++)
    {
        int jn = 0;
        for(int j = 0; j < p; j ++)
        {
            res[i] -= A[jn + i] * R[j];
            jn += n;
        }
    }
}
void jacTres(const double *A, const double *R, const int n, const int p, const int idx, double *res)
{
    for(int i = 0; i < n; i ++)
    {
        int jn = 0;
        for(int j = 0; j < p; j ++)
        {
            res[i + idx] -= A[jn + i] * R[j];
            jn += n;
        }
    }
}
void hessianDviTwo(const int n, const int m, const int idx_i, const int idx_j, const int sz,  double *H)
{
    for(int i = idx_i; i < idx_i + n; i ++)
    {
        for(int j = idx_j; j < idx_j + m; j ++)
        {
            H[i * sz + j] /= 2.0;
        }
    }
}
void multiMatrix(const double *A, const double *B, const int n, const int m, const int p, double *res)
{
    for(int i = 0; i < n; i ++)
    {
        for(int k = 0; k < p; k ++)
        {
            for(int j = 0; j < m; j ++)
            {
                res[i * m + j] += A[i * p + k] * B[k * m + j];
            }
        }
    }
}  
void transpose(const double *A, const int n, const int m, double *res)
{
    for(int i = 0; i < n; i ++)
    {
        for(int j = 0; j < m; j ++)
        {
            res[j * n + i] = A[i * m + j];
        }
    }
}
void printMatrix(const double *A, const int n, const int m)
{
    for(int i = 0; i < n; i ++)
    {
        for(int j = 0; j < m; j ++)
        {
            std::cout << A[i * m + j] << "  ";
        }
        std::cout << std::endl;
    }
}
void numMulMatrix(double *A, const int n, const int m, double x)
{
    for(int i = 0; i < n * m; i ++) A[i] *= x;
}

void addMatrix(const double *A, const double *B, const int n, const int m, double *C)
{
    for(int i = 0; i < n * m; i ++)
    {
        C[i] = A[i] + B[i];
    }
}
double computeModelChange(const double *delta, const double *b, const double lambda, const int sz)
{
    double ans = 0;
    for(int i = 0; i < sz; i ++)
    {
        ans += 0.5 * (delta[i] * (lambda * delta[i] + b[i]));
    }
    return ans;
}

void block(const double *A, const int index_x, const int index_y, const int sx, const int sy, const int sz, double *b)
{
    int idx = 0;
    int m = index_x * sz;
    for(int i = index_x; i < index_x + sx; i ++)
    {
        for(int j = index_y; j < index_y + sy; j ++)
        {
            b[idx ++] = A[m + j];
        }
        m += sz;
    }
}
void setBlock(double *A, const int index_x, const int index_y, const int sx, const int sy, const int sz, const double *b)
{
    int idx = 0;
    int m = index_x * sz;
    for(int i = index_x; i < index_x + sx; i ++)
    {
        for(int j = index_y; j < index_y + sy; j ++)
        {
            A[m + j] = b[idx ++];
        }
        m += sz;
    }
}

void setMultiBlock(const double *A, const double *B, const int index_x, const int index_y, const int sz, double *C)
{
    for(int i = index_x; i < index_x + 6; i ++)
    {
        for(int k = index_y; k < index_y + 3; k ++)
        {
            for(int j = index_y; j < index_y + 3; j ++)
            {
                C[i * sz + j] += A[i * sz + k] * B[k * sz + j];
                
            }

        }
    }
}

void yawPitchRollToRotationMatrix(const double yaw, const double pitch, const double roll, double R[9])
{
    double y = yaw / (180.0) * (M_PI);
    double p = pitch / (180.0) * (M_PI);
    double r = roll / (180.0) * (M_PI);

    R[0] = cos(y) * cos(p);
    R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
    R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
    R[3] = sin(y) * cos(p);
    R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
    R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
    R[6] = -sin(p);
    R[7] = cos(p) * sin(r);
    R[8] = cos(p) * cos(r);
}
double normalizeAngle(const double &angle_degrees)
{
    if (angle_degrees > (180.0))
        return angle_degrees - (360.0);
    else if (angle_degrees < (-180.0))
        return angle_degrees + (360.0);
    else
        return angle_degrees;
}

void rotationMatrixTranpose(const double R[9], double inv_R[9])
{
    inv_R[0] = R[0];
    inv_R[1] = R[3];
    inv_R[2] = R[6];
    inv_R[3] = R[1];
    inv_R[4] = R[4];
    inv_R[5] = R[7];
    inv_R[6] = R[2];
    inv_R[7] = R[5];
    inv_R[8] = R[8];
}
void rotationMatrixRotatePoint(const double R[9], const double t[9], double r_t[9])
{
    r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
    r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
    r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
}

}

