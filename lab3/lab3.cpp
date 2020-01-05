#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <iostream>

using namespace std;
const int root = 0;

void printMatrix(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << matrix[i, j] << '\t';
		}
		cout << endl;
	}
}


int main(int argc, char *argv[])
{
	int rank, size, M, N, workPerProc, extraWork, myRowsSize;
	double endTime, startTime;
	int* matrix = NULL;
	int* vector = NULL;
	int* result = NULL;
	int* sendcounts = NULL;
	int* senddispls = NULL;
	int* recvcounts = NULL;
	int* recvdispls = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == root)
	{
		cout << "Matrix multiplication by vector" << endl << "Process count = " << size << endl;
		cout << "Enter the number of matrix rows:" << endl;
		cin >> M;
		if (M < 1)	return EXIT_FAILURE;
		cout << "Enter the number of matrix columns:" << endl << endl;
		cin >> N;
		if (N < 1)	return EXIT_FAILURE;

		matrix = new int[M * N];
		
		cout << "matrix:" << endl;
		for (int i = 0; i < M; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				matrix[N * i + j] = rand() % 100;
				cout << matrix[N * i + j] << '\t';
			}
			cout << endl;
		}
		cout << endl;

		vector = new int[N];
		// generate vector
		cout << "vector:" << endl;
		for (int i = 0; i < N; ++i)
		{
			vector[i] = rand() & 100;
			cout << vector[i]  << endl;
		}

		cout << endl;
		startTime = MPI_Wtime();
		sendcounts = new int[size];
		senddispls = new int[size];
		recvcounts = new int[size];
		recvdispls = new int[size];

		workPerProc = M / size;
		extraWork = M - workPerProc * size;
		int prefixSum = 0;
		for (int i = 0; i < size; ++i)
		{
			recvcounts[i] = (i < extraWork) ? workPerProc + 1 : workPerProc;
			sendcounts[i] = recvcounts[i] * N;
			recvdispls[i] = prefixSum;
			senddispls[i] = prefixSum * N;
			prefixSum += recvcounts[i];
		}
	}

	MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (0 != rank)
		vector = new int[N];
	MPI_Bcast(vector, N, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank != root)
	{
		workPerProc = M / size;
		extraWork = M % size;
	}
	myRowsSize = rank < extraWork ? workPerProc + 1 : workPerProc;
	int* matrixPart = new int[myRowsSize * N];
	MPI_Scatterv(matrix, sendcounts, senddispls, MPI_INT, matrixPart, myRowsSize * N, MPI_INT, 0, MPI_COMM_WORLD);
	if (0 == rank)
	{
		delete[] sendcounts;
		delete[] senddispls;
		delete[] matrix;
	}

	int* resultPart = new int[myRowsSize];

	for (int i = 0; i < myRowsSize; ++i)
	{
		resultPart[i] = 0;
		for (int j = 0; j < N; ++j)
		{
			resultPart[i] += matrixPart[i * N + j] * vector[j];
		}
	}
	delete[] matrixPart;
	delete[] vector;

	if (rank == root)
		result = new int[M];
	MPI_Gatherv(resultPart, myRowsSize, MPI_INT, result, recvcounts, recvdispls, MPI_INT, 0, MPI_COMM_WORLD);
	delete[] resultPart;
	if (rank == root)
	{
		endTime = MPI_Wtime();
		delete[] recvcounts;
		delete[] recvdispls;
		cout << "result:" << endl;
		//for (int i = 0; i < M; ++i)
		//	cout << result[i] <<endl;
		printMatrix(M, 1, result);


		cout <<endl<< "Time elapsed: " << (endTime - startTime) * 1000 << "ms\n" << endl << endl;
		cout << endl;
		delete[] result;
	}
	MPI_Finalize();
	return 0;
}
