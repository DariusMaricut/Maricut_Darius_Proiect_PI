#ifndef PROIECT1_H
#define PROIECT1_H

#include <opencv2/opencv.hpp>
using namespace cv;

// Convertim orice imagine color la grayscale
void convertToGray(const Mat& color, Mat& gray);

// Detectam muchiile cu un filtru de gradient + prag
void detectEdges(const Mat& gray, Mat& edges, int gradThresh);

// Construim accumulatorul Hough (ro, teta) si desenam liniile direct

Mat houghAndDraw(const Mat& edges, const Mat& gray, Mat& outputColor, int votesThresh);

#endif // PROIECT1_H