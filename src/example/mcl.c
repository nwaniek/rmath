#include <example/mcl.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

void
mcl_expand(matf *M, int a)
{
	assert(M != NULL);

	matf *temp = matf_cpy(M);
	for (int i = 0; i < a-1; i++) {
		matf *temp2 = matf_mul(temp, M);
		matf_free(temp);
		temp = temp2;
	}
	matf_cpyi(M, temp);
	matf_free(temp);
}

void
mcl_inflate(matf *M, float a)
{
	assert(M != NULL);
	assert(a > 1.0f);

	// can't use applyi (due to parameter a)
	for (size_t c = 0; c < M->cols; c++) {
		for (size_t r = 0; r < M->rows; r++) {
			float v = powf(MIDX(M, r, c), a);
			MIDX(M, r, c) = v;
		}

		matf_norm_c1(M, c);
	}
}

void
mcl_add_loops(matf *M)
{
	assert(M != NULL);
	assert(M->cols == M->rows);

	for (size_t r = 0; r < M->rows; r++)
		MIDX(M,r,r) = 1.0f;
}

void
mcl_remove_loops(matf *M)
{
	assert(M != NULL);
	assert(M->cols == M->rows);

	for (size_t r = 0; r < M->rows; r++)
		MIDX(M,r,r) = 0.0f;
}


void
mcl_demo()
{
	matf *A = matf_new(5, 5);

	// setup / prepare
	MIDX(A,0,0) = 0.00f;
	MIDX(A,0,1) = 0.35f;
	MIDX(A,0,2) = 0.10f;
	MIDX(A,0,3) = 0.15f;
	MIDX(A,0,4) = 0.40f;

	MIDX(A,1,0) = 0.35f;
	MIDX(A,1,1) = 0.00f;
	MIDX(A,1,2) = 0.35f;
	MIDX(A,1,3) = 0.00f;
	MIDX(A,1,4) = 0.30f;

	MIDX(A,2,0) = 0.10f;
	MIDX(A,2,1) = 0.35f;
	MIDX(A,2,2) = 0.00f;
	MIDX(A,2,3) = 0.55f;
	MIDX(A,2,4) = 0.00f;

	MIDX(A,3,0) = 0.15f;
	MIDX(A,3,1) = 0.00f;
	MIDX(A,3,2) = 0.55f;
	MIDX(A,3,3) = 0.00f;
	MIDX(A,3,4) = 0.30f;

	MIDX(A,4,0) = 0.40f;
	MIDX(A,4,1) = 0.30f;
	MIDX(A,4,2) = 0.00f;
	MIDX(A,4,3) = 0.30f;
	MIDX(A,4,4) = 0.00f;

	// real demo starts here
	// FIXME: remove fixed number of iterations, make it auto-adapt
	// to difference between previous step and current
	for (size_t i = 0; i < 20; i++) {
		mcl_add_loops(A);
		matf_norm_c(A);

		mcl_expand(A, 2);
		mcl_inflate(A, 2.0f);

		mcl_remove_loops(A);
		matf_norm_c(A);

		matf_dump(A); printf("\n");
	}
}
