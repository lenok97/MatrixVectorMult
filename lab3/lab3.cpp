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
			cout << matrix[n * i + j] << '\t';
		}
		cout << endl;
	}
	cout << endl;
}

void generateSimpleMatrix(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
		{
			matrix[n * i + j]=i;
		}
}

void generateSimpleVector(int n, int* vector)
{
	for (int i = 0; i < n; i++)
		vector[i] = 1;
}

void generateRandomMatrix(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
		{
			matrix[n * i + j] =  rand() % 100;
		}
}

int main(int argc, char *argv[])
{

	int rank, size;
	double endTime, startTime;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	while (true)
	{
		int* matrix = NULL;
		int* vector = NULL;
		int* result = NULL;
		int n, workPerProc, extraWork, myRowsSize;

		int* sendcounts = new int[size];
		int* senddispls = new int[size];
		int* recvcounts = new int[size];
		int* recvdispls = new int[size];

		if (rank == root)
		{
			cout << "Matrix multiplication by vector" << endl << "Process count = " << size << endl;
			cout << "Enter n:" << endl;
			cin >> n;

			matrix = new int[n*n];

			cout << "matrix:" << endl;
			generateSimpleMatrix(n, n, matrix);
			//generateRandomMatrix(N, M, matrix);
			if (n<100)
				printMatrix(n, n, matrix);

			vector = new int[n];
			generateSimpleVector(n, vector);
			//generateRandomMatrix(1, N, vector);
			cout << "vector:" << endl;
			if (n < 100)
			printMatrix(1, n, vector);
			startTime = MPI_Wtime();

			workPerProc = n / size;
			extraWork = n - workPerProc * size;
			int prefixSum = 0;
			for (int i = 0; i < size; ++i)
			{
				recvcounts[i] = (i < extraWork) ? workPerProc + 1 : workPerProc;
				sendcounts[i] = recvcounts[i] * n;
				recvdispls[i] = prefixSum;
				senddispls[i] = prefixSum * n;
				prefixSum += recvcounts[i];
			}
		}

		MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (rank != root)
			vector = new int[n];
		MPI_Bcast(vector,n, MPI_INT, 0, MPI_COMM_WORLD);

		workPerProc = n / size;
		extraWork = n % size;
		myRowsSize = rank < extraWork ? workPerProc + 1 : workPerProc;
		int* matrixPart = new int[myRowsSize * n];

		// разбивает сообщение из буфера посылки процесса root на части
		//Начало расположения элементов блока, посылаемого i - му процессу, 
		//задается в массиве смещений displs, а число посылаемых элементов - в массиве sendcounts
		MPI_Scatterv(matrix, sendcounts, senddispls, MPI_INT, matrixPart, myRowsSize * n, MPI_INT, root, MPI_COMM_WORLD);
		if (rank == root)
		{
			delete[] sendcounts;
			delete[] senddispls;
			delete[] matrix;
		}

		int* resultPart = new int[myRowsSize];

		for (int i = 0; i < myRowsSize; ++i)
		{
			resultPart[i] = 0;
			for (int j = 0; j < n; ++j)
			{
				resultPart[i] += matrixPart[i *n + j] * vector[j];
			}
		}
		delete[] matrixPart;
		delete[] vector;

		if (rank == root)
			result = new int[n];
		//собирает блоки с разным числом элементов от каждого процесса, 
		//количество элементов, принимаемых от каждого процесса, задается в массиве recvcounts.
		//Эта функция обеспечивает также большую гибкость при размещении данных в процессе - получателе, 
		//благодаря введению в качестве параметра массива смещений displs.
		MPI_Gatherv(resultPart, myRowsSize, MPI_INT, result, recvcounts, recvdispls, MPI_INT, root, MPI_COMM_WORLD);
		delete[] resultPart;
		if (rank == root)
		{
			endTime = MPI_Wtime();
			delete[] recvcounts;
			delete[] recvdispls;
			cout << "result:" << endl;
			printMatrix(1, n, result);
			delete[] result;
			cout << endl << "Time elapsed: " << (endTime - startTime) * 1000 << "ms\n" << endl;
		}

	}
	MPI_Finalize();
	return 0;
}
