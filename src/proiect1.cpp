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

// Muchii prin algoritm Canny
void detectEdges(const Mat& gray, Mat& edges, int gradThresh) {
    int rows = gray.rows, cols = gray.cols;
    Mat temp = Mat(rows, cols, CV_8UC1);
    edges = Mat(rows, cols, CV_8UC1, Scalar(0));

    // Gaussian blur pentru reducerea zgomotului
    for(int i = 1; i < rows-1; i++) {
        for(int j = 1; j < cols-1; j++) {
            int sum = 0;
            for(int di = -1; di <= 1; di++) {
                for(int dj = -1; dj <= 1; dj++) {
                    sum += gray.at<uchar>(i+di, j+dj);
                }
            }
            temp.at<uchar>(i,j) = sum / 9;
        }
    }

    // Calculam gradientul folosind Sobel
    Mat gradX = Mat(rows, cols, CV_8UC1);
    Mat gradY = Mat(rows, cols, CV_8UC1);
    Mat gradMag = Mat(rows, cols, CV_8UC1);
    Mat gradDir = Mat(rows, cols, CV_8UC1);

    for(int i = 1; i < rows-1; i++) {
        for(int j = 1; j < cols-1; j++) {

            int gx = -temp.at<uchar>(i-1,j-1) + temp.at<uchar>(i-1,j+1) +
                    -2*temp.at<uchar>(i,j-1) + 2*temp.at<uchar>(i,j+1) +
                    -temp.at<uchar>(i+1,j-1) + temp.at<uchar>(i+1,j+1);


            int gy = -temp.at<uchar>(i-1,j-1) - 2*temp.at<uchar>(i-1,j) - temp.at<uchar>(i-1,j+1) +
                    temp.at<uchar>(i+1,j-1) + 2*temp.at<uchar>(i+1,j) + temp.at<uchar>(i+1,j+1);

            gradX.at<uchar>(i,j) = abs(gx) / 4;
            gradY.at<uchar>(i,j) = abs(gy) / 4;
            gradMag.at<uchar>(i,j) = (abs(gx) + abs(gy)) / 4;

            // Calculam directia gradientului (0-180 grade)
            float angle = atan2(gy, gx) * 180 / CV_PI;
            if(angle < 0) angle += 180;
            gradDir.at<uchar>(i,j) = uchar(angle);
        }
    }

    // Non-maximum suppression
    Mat nms = Mat(rows, cols, CV_8UC1, Scalar(0));
    for(int i = 1; i < rows-1; i++) {
        for(int j = 1; j < cols-1; j++) {
            int mag = gradMag.at<uchar>(i,j);
            if(mag < gradThresh) continue;

            float angle = gradDir.at<uchar>(i,j);

            // Determinam directia de verificare
            int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
            if((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
                dx1 = 0; dy1 = 1; dx2 = 0; dy2 = -1;
            }
            else if(angle >= 22.5 && angle < 67.5) {
                dx1 = 1; dy1 = 1; dx2 = -1; dy2 = -1;
            }
            else if(angle >= 67.5 && angle < 112.5) {
                dx1 = 1; dy1 = 0; dx2 = -1; dy2 = 0;
            }
            else {
                dx1 = 1; dy1 = -1; dx2 = -1; dy2 = 1;
            }

            // Verificam vecinii in directia gradientului
            int mag1 = gradMag.at<uchar>(i+dy1, j+dx1);
            int mag2 = gradMag.at<uchar>(i+dy2, j+dx2);

            if(mag >= mag1 && mag >= mag2) {
                nms.at<uchar>(i,j) = 255;
            }
        }
    }

    // Histerezis cu doua praguri
    int highThresh = gradThresh;
    int lowThresh = gradThresh * 0.4;

    // Prima trecere - marcam muchiile puternice
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(nms.at<uchar>(i,j) == 255) {
                if(gradMag.at<uchar>(i,j) >= highThresh) {
                    edges.at<uchar>(i,j) = 255;
                }
            }
        }
    }

    // A doua trecere - extindem muchiile
    bool changed;
    int maxIterations = 4;
    int iteration = 0;
    do {
        changed = false;
        for(int i = 1; i < rows-1; i++) {
            for(int j = 1; j < cols-1; j++) {
                if(edges.at<uchar>(i,j) == 0 && nms.at<uchar>(i,j) == 255) {
                    if(gradMag.at<uchar>(i,j) >= lowThresh) {
                        // Verificam daca are vecini puternici
                        for(int di = -1; di <= 1; di++) {
                            for(int dj = -1; dj <= 1; dj++) {
                                if(edges.at<uchar>(i+di, j+dj) == 255) {
                                    edges.at<uchar>(i,j) = 255;
                                    changed = true;
                                    break;
                                }
                            }
                            if(changed) break;
                        }
                    }
                }
            }
        }
        iteration++;
    } while(changed && iteration < maxIterations);

    // Eliminam doar muchiile de pe marginea exterioara
    int margin = 2;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(i < margin || j < margin || i >= rows-margin || j >= cols-margin) {
                edges.at<uchar>(i,j) = 0;
            }
        }
    }
}

// Hough: construim segmente de linii pe edge-uri
Mat houghAndDraw(const Mat& edges, const Mat& gray, Mat& outputColor, int votesThresh) {
    int rows = edges.rows, cols = edges.cols;

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

    // Normalizam si cream imaginea spatiului Hough
    Mat houghSpace(nrho, nteta, CV_8UC1, Scalar(0));
    int maxVotes = 0;
    for (int i = 0; i < total; i++) {
        if (acc[i] > maxVotes) maxVotes = acc[i];
    }

    for (int r = 0; r < nrho; r++) {
        for (int t = 0; t < nteta; t++) {
            int val = acc[r * nteta + t];
            if (val > 0) {
                double normalized = 255.0 * log(1 + val) / log(1 + maxVotes);
                houghSpace.at<uchar>(r, t) = uchar(normalized);
            }
        }
    }

    // Desenam liniile detectate
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            if(edges.at<uchar>(y,x) == 255) {
                bool foundLine = false;
                for(int t = 0; t < nteta && !foundLine; t++) {
                    double theta = t * CV_PI / 180.0;
                    double rho = x * cos(theta) + y * sin(theta);
                    int irho = int(rho + diag + 0.5);
                    if(irho >= 0 && irho < nrho) {
                        if(acc[irho * nteta + t] >= votesThresh) {
                            int width = 1;
                            for(int dy = -width; dy <= width; dy++) {
                                for(int dx = -width; dx <= width; dx++) {
                                    int ny = y + dy;
                                    int nx = x + dx;
                                    if(ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
                                        // Verificam daca pixelul vecin este pe aceeasi linie
                                        double rho2 = nx * cos(theta) + ny * sin(theta);
                                        if(abs(rho2 - rho) < 0.5) {
                                            outputColor.at<Vec3b>(ny, nx)[0] = 0;
                                            outputColor.at<Vec3b>(ny, nx)[1] = 0;
                                            outputColor.at<Vec3b>(ny, nx)[2] = 255;
                                        }
                                    }
                                }
                            }
                            foundLine = true;
                        }
                    }
                }
            }
        }
    }

    std::free(acc);
    return houghSpace;
}
