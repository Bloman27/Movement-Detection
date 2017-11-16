
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "opencv2/opencv.hpp"

#define NO_FRAMES_TO_PROCESS 10
#define COLS_TO_PROCESS 32
#define ROWS_TO_PROCESS 32
using namespace cv;


//dev_iamge -> obraz wejœciowy do przetworzenia
//dev_output -> obraz wyjœciowy
//cols -> liczba kolumn (wspolrzedna x)
//rows -> liczba wierszy (wspolrzedna y)
//avgValues -> œrednia wartoœæ pikseli obszarów (z ostanich np. 10 klatek)
//movementDetected true = wykryto ruch, false = brak ruchu
//input_daviation 
//output deviation 
__global__ void cuMovementDetection(unsigned char *dev_image, unsigned char *dev_out, int cols, int rows, float* avgValues, float *input_deviation, bool *movementDetected) {// , , float *output_deviation) {
																													//float *dev_output,
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;

	int n_x_segments = cols / COLS_TO_PROCESS;
	int n_y_segments = rows / ROWS_TO_PROCESS;

	int thread_column = tx % COLS_TO_PROCESS;
	int thread_row = tx / COLS_TO_PROCESS;

	int position = 0;

	int x_start = thread_column * n_x_segments;
	int x_end = (thread_column + 1) * n_x_segments;
	int y_start = thread_row * n_y_segments;
	int y_end = (thread_row + 1) * n_y_segments;

	float avg = 0;

	
	for (int x = x_start; x < x_end; x++) {
		for (int y = y_start; y < y_end; y++) {
			position = x + y* cols;
			avg += dev_image[position];
		}
	}
	
	avg /= (n_x_segments * n_y_segments);

	
	if ((avg > avgValues[tx] + max(input_deviation[tx], 5.0f)) || (avg < avgValues[tx] - max(input_deviation[tx], 5.0f))) {
		
		*movementDetected = true;
	}

	//Update dewiacji
	//avg N-1
	float temp = avg - avgValues[tx];
	//avg N
	avgValues[tx] = avg / NO_FRAMES_TO_PROCESS + (NO_FRAMES_TO_PROCESS - 1) * avgValues[tx] / NO_FRAMES_TO_PROCESS;
	
	float s2 = ((NO_FRAMES_TO_PROCESS - 1)*(input_deviation[tx] * input_deviation[tx]) + (avg - avgValues[tx]) * temp)/NO_FRAMES_TO_PROCESS;

	input_deviation[tx] = sqrt(s2);
	
	__syncthreads();
	

}

bool movementDetectionWithCuda(unsigned char *image, int cols, int rows, float *avgValues, float *deviation, bool *movementDetected) {
	cudaError_t cudaStatus;

	bool *flag = 0;
	unsigned char *dev_input = 0, *dev_output = 0;
	float *avgValues_input = 0, *deviation_input = 0;

	cudaStatus = cudaSetDevice(0);

	//alokacja pamieci na obraz
	cudaStatus = cudaMalloc((void**)&dev_input, cols * rows * sizeof(char));
	//alokacja pamieci na wartosci srednie
	cudaStatus = cudaMalloc((void**)&avgValues_input, COLS_TO_PROCESS * ROWS_TO_PROCESS * sizeof(float));
	//alokacja pamieci na flage wykrycia ruchu
	cudaStatus = cudaMalloc((void**)&flag, sizeof(bool));
	//alokacja pamiêci na dewiacje
	cudaStatus = cudaMalloc((void**)&deviation_input, COLS_TO_PROCESS * ROWS_TO_PROCESS * sizeof(float));

	//przekopiowanie do pamieci gpu
	cudaStatus = cudaMemcpy(dev_input, image, cols * rows * sizeof(char), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(avgValues_input, avgValues, COLS_TO_PROCESS * ROWS_TO_PROCESS * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(flag, movementDetected, sizeof(bool), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(deviation_input, deviation, COLS_TO_PROCESS * ROWS_TO_PROCESS * sizeof(float), cudaMemcpyHostToDevice);

//	dim3 dimGrid(cols / COLS_TO_PROCESS, rows / ROWS_TO_PROCESS);
	//dim3 dimBlock(COLS_TO_PROCESS, ROWS_TO_PROCESS);

	dim3 dimGrid(1);
	dim3 dimBlock(COLS_TO_PROCESS* ROWS_TO_PROCESS);

	cuMovementDetection <<<dimGrid, dimBlock >>> (dev_input, dev_output, cols, rows, avgValues_input, deviation_input, flag); // , input_Deviation, output_Deviation);
																									//dev_output,
	cudaStatus = cudaDeviceSynchronize();
	cudaStatus = cudaMemcpy(image, dev_input, cols * rows * sizeof(char), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(avgValues, avgValues_input, COLS_TO_PROCESS * ROWS_TO_PROCESS * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(movementDetected, flag, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(deviation, deviation_input, COLS_TO_PROCESS * ROWS_TO_PROCESS * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_input);
	cudaFree(flag);
	cudaFree(avgValues_input);
	cudaFree(deviation_input);

	return cudaStatus;

}



int main(int, char**)
{
	VideoCapture camera(0); // open the default camera
	if (!camera.isOpened())  // check if we succeeded
		return -1;

	bool movementDetected = false;

	namedWindow("video", 1);

	//int tempAvgValueOfFrames[ROWS_TO_PROCESS][COLS_TO_PROCESS];
	//float tempDeviation[ROWS_TO_PROCESS][COLS_TO_PROCESS];

	//Zawiera œredni¹ wartoœæ klatek dla ostatnich klatek
	float avgAreaValue[NO_FRAMES_TO_PROCESS][ROWS_TO_PROCESS][COLS_TO_PROCESS];


	//Wyliczenie œredniej wartoœci pikseli dla ka¿dego z obszarów dla ka¿dej klatki
	//for each frame
	for (int k = 0; k < NO_FRAMES_TO_PROCESS; k++) {

		//input
		Mat frame;
		camera >> frame; // get a new frame from camera

						 //number of pixels in row and column
		int rows = frame.rows;
		int cols = frame.cols;


		if (!frame.isContinuous()) break;

		//rgb has 3 channels so No. of columns is 3 times greater
		cols *= frame.channels();
		// for each "BIG" row
		for (int r = 1; r <= ROWS_TO_PROCESS; r++) {
			//for each "BIG" column
			for (int c = 1; c <= COLS_TO_PROCESS; c++) {

				uchar* pFrame;
				int avgValue = 0;;
				//row of pixels
				for (int y_row = rows / ROWS_TO_PROCESS * (r - 1); y_row < rows / ROWS_TO_PROCESS * r; ++y_row)
				{
					pFrame = frame.ptr<uchar>(y_row);
					//each pixel in row
					for (int x_col = cols / COLS_TO_PROCESS * (c - 1); x_col < cols / COLS_TO_PROCESS * c; ++x_col)
					{
						avgValue += pFrame[x_col];
					}
				}

				avgAreaValue[k][r - 1][c - 1] = avgValue / (cols / COLS_TO_PROCESS * rows / ROWS_TO_PROCESS);
				//std::cout << k << "  " << avgAreaValue[k][r - 1][c - 1] << std::endl;
			}
		}


		imshow("video", frame);
		//imshow("modified", modified_frame);

		if (waitKey(25) != 255) return -1;

	}


	//avg value of frames
	float avgValueOfFrames[ROWS_TO_PROCESS][COLS_TO_PROCESS];

	//Wyliczenie œredniej wartoœci pikseli z kilku klatek w danym obszarze
	for (int r = 0; r < ROWS_TO_PROCESS; r++) {
		for (int c = 0; c < COLS_TO_PROCESS; c++) {
			avgValueOfFrames[r][c] = 0;
			for (int f = 0; f < NO_FRAMES_TO_PROCESS; f++) {
				avgValueOfFrames[r][c] += avgAreaValue[f][r][c];
			}
			avgValueOfFrames[r][c] /= NO_FRAMES_TO_PROCESS;
			std::cout << avgValueOfFrames[r][c] << std::endl;
		}
	}


	//obliczenie odchylenia standardowego dla ka¿dego z obszarów
	float deviation[ROWS_TO_PROCESS][COLS_TO_PROCESS];

	for (int r = 0; r < ROWS_TO_PROCESS; r++) {
		for (int c = 0; c < COLS_TO_PROCESS; c++) {
			deviation[r][c] = 0;
			for (int f = 0; f < NO_FRAMES_TO_PROCESS; f++) {
				deviation[r][c] += avgAreaValue[f][r][c] * avgAreaValue[f][r][c] - avgValueOfFrames[r][c] * avgValueOfFrames[r][c];
			}
			deviation[r][c] /= NO_FRAMES_TO_PROCESS;
			deviation[r][c] = sqrt(deviation[r][c]);
			//std::cout << "dev  " << deviation[r][c] << std::endl;
		}
	}

	int frameToChange = -1;

	float avgValueOfFrames2[ROWS_TO_PROCESS * COLS_TO_PROCESS];
	float deviation2[ROWS_TO_PROCESS * COLS_TO_PROCESS];
	for (int i = 0; i < ROWS_TO_PROCESS; i++)
		for (int j = 0; j < COLS_TO_PROCESS; j++) {
			avgValueOfFrames2[j + i * COLS_TO_PROCESS] = avgValueOfFrames[i][j];
			deviation2[j + i * COLS_TO_PROCESS] = deviation[i][j];
		}

	int counter = 0;

	for (;;)
	{
		
		movementDetected = false;
		Mat frame;
		camera >> frame;

		imshow("video", frame);
		//imshow("modified", modified_frame);

		if (waitKey(25) != 255) return -1;

		//number of pixels in row and column
		int rows = frame.rows;
		int cols = frame.cols;

		if (!frame.isContinuous()) break;

		//rgb has 3 channels so No. of columns is 3 times greater
		cols *= frame.channels();

		unsigned char *ptrFrame = frame.data;

		movementDetectionWithCuda(ptrFrame, cols, rows, avgValueOfFrames2, deviation2, &movementDetected);

		if (movementDetected) {
			std::cout << "alert " << counter << std::endl;
			counter++;
		}

		//for (int i = 0; i < 9; i++) {
			//std::cout << deviation2[i] << std::endl;
		//}
		//imshow("video", frame);
		//waitKey(0);


	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}

