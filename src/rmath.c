#include <rmath.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>

matf*
matf_new(size_t rows, size_t cols)
{
	matf *C = malloc(sizeof(matf));
	C->rows = rows;
	C->cols = cols;
	C->v = calloc(rows * cols, sizeof(float));
	return C;
}

matf*
matf_cpy(matf *M)
{
	matf *R = matf_new(M->rows, M->cols);
	memcpy(R->v, M->v, M->rows * M->cols * sizeof(float));
	return R;
}

void
matf_cpyi(matf *Dst, matf *Src)
{
	assert(Dst != NULL && Src != NULL);
	assert(Dst->rows == Src->rows && Dst->cols == Src->cols);
	memcpy(Dst->v, Src->v, Src->rows * Src->cols * sizeof(float));
}

matf*
matf_new_val(size_t rows, size_t cols, float v)
{
	matf *C = matf_new(rows, cols);
	for (size_t i = 0; i < rows * cols; C->v[i] = v, i++);
	return C;
}

matf*
matf_eye(size_t rows)
{
	matf *C = matf_new(rows, rows);
	for (size_t i = 0; i < rows; i++)
		C->v[M_IDX(i,i,rows)] = 1;
	return C;
}


void
matf_norm_r1(matf *M, size_t r)
{
	float sum = 0.0f;
	for (size_t c = 0; c < M->cols; c++)
		sum += MIDX(M, r, c);
	if (sum == 0.0f) return;
	for (size_t c = 0; c < M->cols; c++)
		MIDX(M, r, c) /= sum;
}

void
matf_norm_r(matf *M)
{
	assert(M != NULL);
	for (size_t r = 0; r < M->rows; r++)
		matf_norm_r1(M, r);
}

void
matf_norm_c1(matf *M, size_t c)
{
	float sum = 0.0f;
	for (size_t r = 0; r < M->rows; r++)
		sum += MIDX(M, r, c);
	if (sum == 0.0f) return;
	for (size_t r = 0; r < M->rows; r++)
		MIDX(M, r, c) /= sum;
}

void
matf_norm_c(matf *M)
{
	assert(M != NULL);
	for (size_t c = 0; c < M->rows; c++)
		matf_norm_c1(M, c);
}

void
matf_set(matf *A, float v)
{
	assert(A != NULL);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		A->v[i] = v;
}

void
matf_free(matf *A)
{
	if (!A) return;
	free(A->v);
	free(A);
}

float
dotp(float *p1, int s1, float *p2, int s2, unsigned int n)
{
	float s = 0.0f;
	for (size_t i = 0; i < n; i++, p1 += s1, p2 += s2)
		s += p1[0] * p2[0];
	return s;
}

matf*
matf_apply(matf *A, float (*fp)(float))
{
	assert(A != NULL && fp != NULL);
	matf *C = matf_new(A->rows, A->cols);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		C->v[i] = fp(A->v[i]);
	return C;
}

void
matf_applyi(matf *A, float (*fp)(float))
{
	assert(A != NULL && fp != NULL);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		A->v[i] = fp(A->v[i]);
}

matf*
matf_add(matf *A, matf *B)
{
	assert(A->cols == B->cols && A->rows == B->rows);
	matf *C = matf_new(A->rows, B->rows);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		C->v[i] = A->v[i] + B->v[i];
	return C;
}

void
matf_addi(matf *A, matf *B)
{
	assert(A->cols == B->cols && A->rows == B->rows);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		A->v[i] += B->v[i];
}

matf*
matf_sub(matf *A, matf *B)
{
	assert(A->cols == B->cols && A->rows == B->rows);
	matf *C = matf_new(A->rows, B->rows);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		C->v[i] = A->v[i] - B->v[i];
	return C;
}

void
matf_subi(matf *A, matf *B)
{
	assert(A->cols == B->cols && A->rows == B->rows);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		A->v[i] -= B->v[i];
}

matf*
matf_mul(matf *A, matf *B)
{
	assert(A != NULL && B != NULL);
	assert(A->cols == B->rows);

	matf *C = matf_new(A->rows, B->cols);
	for (size_t r = 0; r < C->rows; r++) {
		for (size_t c = 0; c < C->cols; c++) {
			MIDX(C, r, c) =
				dotp(&A->v[r * A->cols], 1,
				     &B->v[c], B->cols,
				     B->rows);
		}
	}
	return C;
}

inline static float
matf_transposed_v(matf *M, size_t r, size_t c)
{
	assert(M != NULL);
	return M->v[M_IDX_T(r, c, M->cols)];
}

matf*
matf_transpose(matf *A)
{
	assert(A != NULL);
	matf *B = matf_new(A->cols, A->rows);
	for (size_t r = 0; r < A->cols; r++)
		for (size_t c = 0; c < A->rows; c++)
			MIDX(B, r, c) = matf_transposed_v(A, r, c);
	return B;
}

matf*
matf_mul_elems(matf *A, matf *B)
{
	assert(A != NULL && B != NULL && A->rows == B->rows && A->cols == B->cols);
	matf *C = matf_new(A->rows, A->cols);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		C->v[i] = A->v[i] * B->v[i];
	return C;
}

void
matf_mul_elemsi(matf *A, matf *B)
{
	assert(A->rows == B->rows && A->cols == B->cols);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		A->v[i] *= B->v[i];
}

matf*
matf_mul_scalar(matf *A, float s)
{
	assert(A != NULL);
	matf *C = matf_new(A->rows, A->cols);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		C->v[i] = A->v[i] * s;
	return C;
}

void
matf_mul_scalari(matf *A, float s)
{
	assert(A != NULL);
	for (size_t i = 0; i < A->rows * A->cols; i++)
		A->v[i] *= s;
}

matf*
matf_diag(matf *A)
{
	assert(A != NULL && A->cols == A->rows);
	matf *C = matf_new(A->rows, 1);
	for (size_t i = 0; i < A->rows; i++)
		C->v[i] = MIDX(A, i, i);
	return C;
}

void
matf_dump_linear(matf *A)
{
	printf("[ ");
	for (size_t r = 0; r < A->rows; r++) {
		for (size_t c = 0; c < A->cols; c++) {
			printf("%.3f", MIDX(A, r, c));
			if (c != A->cols - 1)
				printf(", ");
		}
		if (r != A->rows - 1)
			printf("; ");
	}
	printf("]\n");
}

void
matf_dump(matf *A)
{
	printf("[ ");
	for (size_t r = 0; r < A->rows; r++) {
		for (size_t c = 0; c < A->cols; c++)
			printf("%.3f ", MIDX(A, r, c));
		if (r != A->rows - 1)
			printf("\n  ");
	}
	printf("]\n");
}

void
matf_dump_transposed(matf *A)
{
	for (size_t r = 0; r < A->cols; r++) {
		for (size_t c = 0; c < A->rows; c++)
			printf("%.3f ", matf_transposed_v(A, r, c));
		printf("\n");
	}
}

float
_randf()
{
	return 10 * (float)rand() / (float)RAND_MAX;
}

matf*
matf_rand(size_t rows, size_t cols)
{
	matf *C = matf_new(rows, cols);
	for (size_t i = 0; i < rows * cols; i++)
		C->v[i] = _randf();
	return C;
}

