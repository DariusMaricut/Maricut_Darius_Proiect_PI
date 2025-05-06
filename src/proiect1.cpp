#include "proiect1.h"
#include <cmath>
#include <cstdlib>

// Grayscale: media celor 3 canale BGR
void convertToGray(const Mat& color, Mat& gray) {
    int rows = color.rows, cols = color.cols;
    gray = Mat(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pix = color.at<Vec3b>(i,j);
            int b = pix[0], g = pix[1], r = pix[2];
            gray.at<uchar>(i,j) = uchar((b + g + r) / 3);
        }
    }
}

// Muchii prin gradient 3x3 + prag
void detectEdges(const Mat& gray, Mat& edges, int gradThresh) {
    int rows = gray.rows, cols = gray.cols;

    edges = Mat(rows, cols, CV_8UC1);
    // Marginile le punem negre
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            if(i == 0 || j == 0 || i == rows - 1 || j == cols - 1) {
                edges.at<uchar>(i,j) = 0;
            } else {
                // calculam Gx si Gy
                int gx = int(gray.at<uchar>(i,j+1)) - int(gray.at<uchar>(i,j-1));
                int gy = int(gray.at<uchar>(i+1,j)) - int(gray.at<uchar>(i-1,j));
                int mag = abs(gx) + abs(gy);
                edges.at<uchar>(i,j) = (mag > gradThresh ? 255 : 0);
            }
        }
    }
}

// Hough: construim segmente de linii pe edge-uri
Mat houghAndDraw(const Mat& edges, const Mat& gray, Mat& outputColor, int votesThresh) {
    int rows = edges.rows, cols = edges.cols;

    // outputColor plecand de la GRAYSCALE
    outputColor = Mat(rows, cols, CV_8UC3);
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            uchar v = gray.at<uchar>(y,x);
            outputColor.at<Vec3b>(y,x)[0] = v;
            outputColor.at<Vec3b>(y,x)[1] = v;
            outputColor.at<Vec3b>(y,x)[2] = v;
        }
    }

    // Parametri Hough
    double diag = std::sqrt(rows * rows + cols * cols);
    int nrho = int(diag * 2) + 1;
    int nteta = 180;

    int total = nrho * nteta;
    int* acc = (int*) std::malloc(total * sizeof(int));
    for (int i = 0; i < total; i++) acc[i] = 0;

    // Votam in accumulator
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (edges.at<uchar>(y,x) == 255) {
                for (int t = 0; t < nteta; t++) {
                    double teta = t * CV_PI / 180.0;
                    double rho = x * cos(teta) + y * sin(teta);
                    int irho = int(rho + diag + 0.5);
                    if (irho >= 0 && irho < nrho) {
                        acc[irho * nteta + t]++;
                    }
                }
            }
        }
    }

    // Cream imaginea spatiului Hough
    Mat houghSpace(nrho, nteta, CV_8UC1, Scalar(0));
    int maxVotes = 0;
    for (int i = 0; i < total; i++) {
        if (acc[i] > maxVotes) maxVotes = acc[i];
    }
    for (int r = 0; r < nrho; r++) {
        for (int t = 0; t < nteta; t++) {
            int val = acc[r * nteta + t];
            houghSpace.at<uchar>(r, t) = uchar(255.0 * val / maxVotes);
        }
    }

    // Coloram pixelii de muchie care au fost votati pentru o linie lunga
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            if(edges.at<uchar>(y,x) == 255) {
                // pentru fiecare unghi, vedem daca pixelul a votat o linie valida
                for(int t = 0; t < nteta; t++) {
                    double theta = t * CV_PI / 180.0;
                    double rho = x * cos(theta) + y * sin(theta);
                    int irho = int(rho + diag + 0.5);
                    if(irho >= 0 && irho < nrho) {
                        if(acc[irho * nteta + t] >= votesThresh) {
                            // coloram pixelul rosu si oprim testele pentru acest pixel
                            outputColor.at<Vec3b>(y,x)[0] = 0;   // B
                            outputColor.at<Vec3b>(y,x)[1] = 0;   // G
                            outputColor.at<Vec3b>(y,x)[2] = 255; // R

                            // Facem linia mai groasa colorand si pixelii vecini
                            int width = 0.75;
                            for(int dy = -width; dy <= width; dy++) {
                                for(int dx = -width; dx <= width; dx++) {
                                    int ny = y + dy;
                                    int nx = x + dx;
                                    if(ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
                                        outputColor.at<Vec3b>(ny, nx)[0] = 0;
                                        outputColor.at<Vec3b>(ny, nx)[1] = 0;
                                        outputColor.at<Vec3b>(ny, nx)[2] = 255;
                                    }
                                }
                            }

                            break;
                        }
                    }
                }
            }
        }
    }

    std::free(acc);
    return houghSpace;
}
