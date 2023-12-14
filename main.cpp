
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>   
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void fftshift(Mat& magnitude)
{
	// Rearrange the quadrants of the Fourier image
	int cx = magnitude.cols / 2;
	int cy = magnitude.rows / 2;

	Mat q0(magnitude, Rect(0, 0, cx, cy));
	Mat q1(magnitude, Rect(cx, 0, cx, cy));
	Mat q2(magnitude, Rect(0, cy, cx, cy));
	Mat q3(magnitude, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

Mat computePowerSpectrum(const Mat& signal)
{
	// Compute autocorrelation function
	Mat autocorrelation;
	matchTemplate(signal, signal, autocorrelation, TM_CCORR_NORMED);

	// Fourier transform of autocorrelation function
	Mat powerSpectrum;
	dft(autocorrelation, powerSpectrum, DFT_COMPLEX_OUTPUT);

	// Split into real and imaginary parts
	Mat planes[2];
	split(powerSpectrum, planes);

	// Calculate magnitude (power spectrum)
	Mat mag;
	magnitude(planes[0], planes[1], mag);

	// Shift the zero frequency component to the center
	fftshift(mag);

	return mag;
}

void calculateFFT(const Mat& inputImage, Mat& outputFFT) {
	Mat padded; // Expand image to optimal size
	int m = getOptimalDFTSize(inputImage.rows);
	int n = getOptimalDFTSize(inputImage.cols);
	copyMakeBorder(inputImage, padded, 0, m - inputImage.rows, 0, n - inputImage.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImage;
	merge(planes, 2, complexImage);

	dft(complexImage, complexImage);

	// Shift the quadrants to center
	fftshift(complexImage);

	// Compute magnitude spectrum
	split(complexImage, planes);
	magnitude(planes[0], planes[1], outputFFT);
}

Mat computeNoisePowerSpectrum(const Mat& noisyImage)
{
	// Convert the input image to double precision
	Mat noisyImageDouble;
	noisyImage.convertTo(noisyImageDouble, CV_64F);

	// Assume noise is in a specific region of the image (you may need to adjust this)
	int noiseWidth = std::min(50, noisyImage.cols - 100);
	int noiseHeight = std::min(50, noisyImage.rows - 100);

	// Ensure that the region is even in size
	noiseWidth -= (noiseWidth % 2 == 1);
	noiseHeight -= (noiseHeight % 2 == 1);

	Rect noiseRegion(100, 100, noiseWidth, noiseHeight);
	Mat noiseRegionImage = noisyImageDouble(noiseRegion).clone();

	// Compute the FFT of the noise region
	Mat noiseRegionFFT;
	calculateFFT(noiseRegionImage, noiseRegionFFT);

	// Calculate the PSD by taking the squared magnitude of the FFT
	Mat psd;
	pow(noiseRegionFFT, 2, psd);

	// Shift the zero frequency component to the center
	fftshift(psd);

	return psd;
}


Mat generateGaussianPSF(Size size, double sigma)
{
	// Calculate the center of the PSF
	Point center(size.width / 2, size.height / 2);

	// Create an empty matrix for the PSF
	Mat psf(size, CV_64F);

	// Generate the Gaussian PSF
	for (int i = 0; i < size.width; ++i)
	{
		for (int j = 0; j < size.height; ++j)
		{
			double x = i - center.x;
			double y = j - center.y;
			psf.at<double>(j, i) = exp(-(x * x + y * y) / (2.0 * sigma * sigma));
		}
	}

	// Normalize the PSF
	normalize(psf, psf, 1.0, NORM_L1);

	return psf;
}

Mat LowLightEnhancment(Mat bgr_image,int val)
{
	Mat lab_image;
	cvtColor(bgr_image, lab_image, COLOR_BGR2Lab);
	Mat grayscale_image;
	cvtColor(bgr_image, grayscale_image, COLOR_BGR2GRAY);

	vector<Mat> lab_planes(3);
	split(lab_image, lab_planes);

	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(val); 
	Mat dst;
	clahe->apply(lab_planes[0], dst);

	dst.copyTo(lab_planes[0]);
	merge(lab_planes, lab_image);

	Mat image_clahe;
	cvtColor(lab_image, image_clahe, COLOR_Lab2BGR);

	return image_clahe;
}

Mat GaussianSharpen(Mat bgr_image)
{
	Mat blurred_image;
	GaussianBlur(bgr_image, blurred_image, Size(5, 5), 2.0);
	Mat unsharpMask = bgr_image - blurred_image;
	Mat sharpened;
	addWeighted(bgr_image, 1.5, unsharpMask, -0.5, 0, sharpened);
	return sharpened;
}

Mat BilateralFiltering(Mat bgr_image)
{
	Mat filteredImage;
	bilateralFilter(bgr_image, filteredImage, 9, 75, 75);
	return filteredImage;
}

int main()
{
	Mat bgr_image = imread("E:\\Downloads\\image3.jpg");
	Mat test_img = imread("E:\\Downloads\\imagedeblur.png");

	int val = 3;
	Mat image_clahe = LowLightEnhancment(bgr_image,val);

	Mat BlurredImage;
	GaussianBlur(test_img, BlurredImage, Size(5, 5), 2.0);

	Mat GaussianSharpener = GaussianSharpen(test_img);

	Mat BilateralSharpener = BilateralFiltering(test_img);
	
	Mat blurred_image;
	GaussianBlur(bgr_image, blurred_image, Size(5, 5), 2.0);
	
	Mat temp = blurred_image;

	Size psfSize(5, 5);
	double sigma = 1.5;
	Mat psf = generateGaussianPSF(psfSize, sigma);

	Mat power_spec = computePowerSpectrum(temp);


	imshow("Original Image", bgr_image);
	imshow("Low Light", image_clahe);
	imshow("Blur", BlurredImage);
	imshow("Sharpened", BilateralSharpener);
	imshow("Deblurred", GaussianSharpener);
	imshow("Original", test_img);


	waitKey();
}
