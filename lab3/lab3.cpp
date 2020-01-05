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
			cout << matrix[n * i + j] << '\t';
		cout << endl;
	}
	cout << endl;
}

void generateSimpleMatrix(int n, int m, int* matrix)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			matrix[n * i + j]=i;
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
			matrix[n * i + j] =  rand() % 100;
}

int main(int argc, char *argv[])
{
	int rank, size;
	double endTime, startTime;
	bool randomMatrix = false;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	while (true)
	{
		int n, workPerProc, extraWork, rowsCount;
		int* matrix = NULL;
		int* vector = NULL;
		int* result = NULL;
		// MPI_Scatterv/MPI_Gatherv params
		int* sendcounts = new int[size]; //количество элементов, принимаемых от каждого процесса
		int* senddispls = new int[size]; //начало расположения элементов блока, посылаемого i-му процессу
		int* recvcounts = new int[size]; //количество элементов, отправляемых каждым процессом
		int* recvdispls = new int[size]; //начало расположения элементов блока, принимаемых от i-ого процесса

		if (rank == root)
		{
			cout << "Matrix multiplication by vector" << endl << "Process count = " << size << endl;
			cout << "Enter n:" << endl;
			cin >> n;

			matrix = new int[n*n];
			vector = new int[n];

			if (randomMatrix)
			{
				generateRandomMatrix(n, n, matrix);
				generateRandomMatrix(1, n, vector);
			}
			else
			{
				generateSimpleMatrix(n, n, matrix);
				generateSimpleVector(n, vector);
			}
			if (n < 100)
			{
				cout << "matrix:" << endl;
				printMatrix(n, n, matrix);
				cout << "vector:" << endl;
				printMatrix(1, n, vector);
			}
			startTime = MPI_Wtime();

			workPerProc = n / size;
			extraWork = n % size;;
			int totalDispl = 0;
			// MPI_Scatterv/MPI_Gatherv params
			for (int i = 0; i < size; i++)
			{
				recvcounts[i] = workPerProc;
				if (i < extraWork)
					recvcounts[i]++; 
				sendcounts[i] = recvcounts[i] * n;
				senddispls[i] = totalDispl * n;
				recvdispls[i] = totalDispl;
				totalDispl += recvcounts[i];
			}
			result = new int[n];
		}

		MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (rank != root)
			vector = new int[n];
		MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);
		// горизонтаальное ленточное разбиение
		workPerProc = n / size;
		extraWork = n % size;
		rowsCount = workPerProc;
		if (rank < extraWork)
			rowsCount++;
		int* matrixPart = new int[rowsCount * n];
		
		// разбивает сообщение из буфера посылки процесса root на части
		MPI_Scatterv(matrix, sendcounts, senddispls, MPI_INT, matrixPart, rowsCount * n, MPI_INT, root, MPI_COMM_WORLD);
		if (rank == root)
		{
			delete[] sendcounts;
			delete[] senddispls;
			delete[] matrix;
		}
		// printMatrix(n, rowsCount, matrixPart);
		int* tempResult = new int[rowsCount];
		for (int i = 0; i < rowsCount; i++)
		{
			tempResult[i] = 0;
			for (int j = 0; j < n; j++)
			{
				tempResult[i] += matrixPart[i *n + j] * vector[j];
			}
		}
		// printMatrix(rowsCount, 1, tempResult);

		// собирает блоки с разным числом элементов от каждого процесса
		MPI_Gatherv(tempResult, rowsCount, MPI_INT, result, recvcounts, recvdispls, MPI_INT, root, MPI_COMM_WORLD);
		delete[] matrixPart;
		delete[] vector;
		delete[] tempResult;

		if (rank == root)
		{
			endTime = MPI_Wtime();
			cout << "result:" << endl;
			printMatrix(1, n, result);
			delete[] recvcounts;
			delete[] recvdispls;
			delete[] result;
			cout << endl << "Time elapsed: " << (endTime - startTime) * 1000 << "ms\n" << endl;
			randomMatrix = !randomMatrix;
		}
	}
	MPI_Finalize();
	return 0;
}
