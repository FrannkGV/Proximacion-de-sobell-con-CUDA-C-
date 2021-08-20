#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string>
#include <time.h>

using namespace std;
using namespace cv;

Mat generateImgColor(uchar4* pixels, int rows, int columns)
{
	Mat frame;
	Mat temp(rows, columns, CV_8UC4, pixels);
	cvtColor(temp, frame, CV_RGBA2BGR);
	return frame;
}


void filterCPU(uchar4* const frameOriginal, uchar4* const frameOutput, int rows, int columns, float* const mask)
{
	int id = 0; // indice central
	int idN = 0; // indice del vecino
	int idMask = 0; // indice de la mascara

	for (int row = 0; row < rows; row++) {
		for (int column = 0; column < columns; column++) {
			id = row * columns + column;

			float X = 0; // primer canal de colores
			float Y = 0; // segundo canal de colores
			float Z = 0; // tercer canal de colores
			float m = 0;

			idMask = 0;

			for (int i = (row - 1); i <= (row + 1); i++) {
				for (int j = (column - 1); j <= (column + 1); j++) {
					if (i >= 0 && j >= 0 && j < columns && i < rows) {
						idN = (i * columns) + j;
						X += frameOriginal[idN].x * mask[idMask];
						Y += frameOriginal[idN].y * mask[idMask];
						Z += frameOriginal[idN].z * mask[idMask];
					}
					else {  // no existe el vecino
						X += frameOriginal[id].x * mask[idMask];
						Y += frameOriginal[id].y * mask[idMask];
						Z += frameOriginal[id].z * mask[idMask];
					}
					idMask++;
				}
			}
			m = max(abs(X), abs(Y));
			m = (max(m, abs(Z)));
			frameOutput[id].x = (uchar)m;
			frameOutput[id].y = (uchar)m;
			frameOutput[id].z = (uchar)m;
			frameOutput[id].w = frameOriginal[id].w;
		}
	}
}

int intDivision(int n, int m) {
	int valor = 0;
	if ((n % m) == 0)
		valor = n / m;
	else
		valor = (n / m) + 1;
	return valor;
}

__device__ int maxGPU(float valor1, float valor2) {
	int maximo = valor1;
	if (valor2 > valor1) maximo = valor2;
	return maximo;
}

__global__ void filterGPU(uchar4* const frameOriginal, uchar4* const frameOutput, int rows, int columns, float* const mask)
{	
	int fila = (blockIdx.x * blockDim.x) + threadIdx.x;
	int columna = (blockIdx.y * blockDim.y) + threadIdx.y;

	//printf("fila: %d , columna: %d \n", fila,columna);

	if (columna >= columns || fila >= rows) {
		return;
	}
	int indice = fila * columns + columna;
	int indVec = 0; // indice del vecino

	float valorX = 0; // primer canal de colores
	float valorY = 0; // segundo canal de colores
	float valorZ = 0; // tercer canal de colores
	float valorMax = 0; // valor máximo de los colores

	int posMask = 0;

	for (int r = (fila - 1); r <= (fila + 1); r++) {
		for (int c = (columna - 1); c <= (columna + 1); c++) {
			if (r >= 0 && c >= 0 && c < columns && r < rows) {  // existe el vecino?
				indVec = (r * columns) + c;
				valorX += frameOriginal[indVec].x * mask[posMask];
				valorY += frameOriginal[indVec].y * mask[posMask];
				valorZ += frameOriginal[indVec].z * mask[posMask];
			}
			else {  // no existe el vecino
				valorX += frameOriginal[indice].x * mask[posMask];
				valorY += frameOriginal[indice].y * mask[posMask];
				valorZ += frameOriginal[indice].z * mask[posMask];
			}
			posMask++;
		}
	}

	valorMax = maxGPU(abs(valorX), abs(valorY));
	valorMax = maxGPU(valorMax, abs(valorZ));
	frameOutput[indice].x = (uchar)valorMax;
	frameOutput[indice].y = (uchar)valorMax;
	frameOutput[indice].z = (uchar)valorMax;
	frameOutput[indice].w = frameOriginal[indice].w;
}

void processCPU() {
	Mat frame;
	Mat frameRGBA;
	Mat frameFinalCPU;
	Mat frameFinalGPU;
	int rows = 0, columns = 0;
	uchar4* frameOriginal, * dframeOriginal;
	uchar4* frameOutputCPU;
	uchar4* frameOutputGPU, *dframeOutputGPU;

	int numBlocks = 0;
	cudaDeviceProp devProp;
	const int maxThreads = 1024;

	//float mask[9] = { -1.0, 0, 1.0, -2.0, 0, 2.0, -1.0, 0, 1.0 };
	//float mask[9] = { -1.0, -1.0, 0, -1.0, 0, 1.0, 0, 1.0, 1.0 }; //sobell
	float mask[9] = { -0.6, -0.6, 0, -0.6, 0, 0.6, 0, 0.6, 0.6 };
	//float mask[9] = { 0, 0, 1.0, 0, 1.0, 0, 1, 0, 0 };
	//float mask[9] = { 0.3, 0.3, 0.3, 0.3, 1, 0.3, 0.3, 0.3, 0.3 };
	float* dmask;

	VideoCapture capture("C:\\Users\\fjgvp\\Videos\\Captures\\videoPrron.mp4");
	namedWindow("Video Original", CV_WINDOW_AUTOSIZE);
	namedWindow("Video CPU", CV_WINDOW_AUTOSIZE);
	namedWindow("Video GPU", CV_WINDOW_AUTOSIZE);

	clock_t timer1 = 0;
	float timerB1 = 0;
	clock_t timer2= 0;
	float timerB2 = 0;

	while (1) {
		capture >> frame;
		if (frame.empty())
			break;

		cvtColor(frame, frameRGBA, CV_BGR2RGBA);

		rows = frame.rows;
		columns = frame.cols;
		const size_t numPixels = rows * columns;

		frameOriginal = (uchar4*)frameRGBA.ptr<unsigned char>(0);
		frameOutputCPU = (uchar4*)malloc(sizeof(uchar4) * numPixels);
		frameOutputGPU = (uchar4*)malloc(sizeof(uchar4) * numPixels);
		memset(frameOutputCPU, 0, sizeof(uchar4) * numPixels);
		memset(frameOutputGPU, 0, sizeof(uchar4) * numPixels);
		
		timer1 = clock();
		filterCPU(frameOriginal, frameOutputCPU, rows, columns, mask);
		frameFinalCPU = generateImgColor(frameOutputCPU, rows, columns);
		timer1 = clock() - timer1;
		timerB1 += timer1;


		//printf("\t\t<<<<< GPU >>>>>\n");

		int maxN = 4; // numero de filas de hilos p/bloque
		int maxM = 4; // numero de columnas de hilos p/bloque

		int numBloquesN = intDivision(rows, maxN);
		int numBloquesM = intDivision(columns, maxM);
		const dim3 dimGrid(numBloquesN, numBloquesM);
		const dim3 dimBlock(maxN, maxM, 1);

		cudaMalloc(&dframeOriginal, sizeof(uchar4) * numPixels);
		cudaMalloc(&dframeOutputGPU, sizeof(uchar4) * numPixels);
		cudaMalloc(&dmask, sizeof(float) * 9);

		cudaMemset(dframeOriginal, 0, sizeof(uchar4) * numPixels);
		cudaMemset(dframeOutputGPU, 0, sizeof(uchar4) * numPixels);

		cudaMemcpy(dframeOriginal, frameOriginal, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
		cudaMemcpy(dmask, mask, sizeof(float) * 9, cudaMemcpyHostToDevice);

		//numBlocks = intDivision(numPixels, maxThreads);
		//printf("Num bloques: %d, num hilos: %d \n", numBlocks, maxThreads);
		//dim3 dimGrid(numBlocks);
		//dim3 dimBlock(maxThreads);

		//printf("filas: %d, columnas: %d", rows, columns);
		timer2 = clock();
		filterGPU << <dimGrid, dimBlock >> > (dframeOriginal, dframeOutputGPU, rows, columns, dmask);
		cudaDeviceSynchronize();
		timer2 = clock() - timer2;
		timerB2 += timer2;

		cudaMemcpy(frameOutputGPU, dframeOutputGPU, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
		frameFinalGPU = generateImgColor(frameOutputGPU, rows, columns);

		imshow("Video CPU", frameFinalCPU);
		imshow("Video GPU", frameFinalGPU);
		imshow("Video Original", frame);
		waitKey(20);
	};

	printf("Operacion en CPU toma %10.3f ms.\n", (((float)timerB1) / CLOCKS_PER_SEC) * 1000);
	printf("Operacion en GPU toma %10.3f ms.\n", (((float)timerB2) / CLOCKS_PER_SEC) * 1000);


	free(frameOriginal);
	free(frameOutputCPU);

	cudaFree(dframeOriginal);
	cudaFree(dframeOutputGPU);
	cudaFree(dmask);

	destroyWindow("Video Original");
	destroyWindow("Video CPU");
	destroyWindow("Video GPU");
}

//void processGPU()
//{
//	VideoCapture capture("C:\\Users\\fjgvp\\Videos\\Captures\\videoPrron.mp4");
//	namedWindow("Video CPU", CV_WINDOW_AUTOSIZE);
//}

int main()
{
	processCPU();
	//processGPU();
	return 0;
}