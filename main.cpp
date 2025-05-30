#include <opencv2/opencv.hpp>
#include "proiect1.h"
#include <cstdio>

using namespace cv;

int main() {
    const char* imagePath = "D:\\Ceva\\An3 Pi2\\proiect1\\src\\images\\lines14.jpg";

    Mat img = imread(imagePath, IMREAD_COLOR);
    if (img.empty()) {
        printf("Nu pot deschide fisierul: %s\n", imagePath);
        return -1;
    }

    // Convertire la grayscale, daca e color
    Mat gray;
    if (img.channels() == 3) {
        convertToGray(img, gray);
    } else {
        gray = img.clone();
    }

    // Extragem muchiile
    Mat edges;
    int gradThresh = 75;
    detectEdges(gray, edges, gradThresh);

    // Transformata Hough + desenare linii peste grayscale
    Mat result;
    int votesThresh = 120;
    Mat houghSpace = houghAndDraw(edges, gray, result, votesThresh);

    imshow("Grayscale",      gray);
    imshow("Edges",          edges);
    imshow("Detected Lines", result);
    imshow("Hough Space",    houghSpace);

    waitKey(0);
    return 0;
}
