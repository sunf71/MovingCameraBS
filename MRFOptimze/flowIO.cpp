// flow_io.cpp
//
// read and write our simple .flo flow file format

// ".flo" file format used for optical flow evaluation
//
// Stores 2-band float image for horizontal (u) and vertical (v) flow components.
// Floats are stored in little-endian order.
// A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
//
//  bytes  contents
//
//  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
//


// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "flowIO.h"

// return whether flow vector is unknown
bool unknown_flow(float u, float v) {
    return (fabs(u) >  UNKNOWN_FLOW_THRESH) 
	|| (fabs(v) >  UNKNOWN_FLOW_THRESH)
	|| _isnan(u) || _isnan(v);
}

bool unknown_flow(float *f) {
    return unknown_flow(f[0], f[1]);
}

// read a flow file into 2-band image
void ReadFlowFile(cv::Mat& img, const char* filename)
{
    if (filename == NULL)
	{
		std::cout<<("ReadFlowFile: empty filename");
		return;
	}

    const char *dot = strrchr(filename, '.');
    if (strcmp(dot, ".flo") != 0)
	{
		std::cout<<("ReadFlowFile (%s): extension .flo expected", filename);
		return;
	}

    FILE *stream = fopen(filename, "rb");
    if (stream == 0)
    {
		std::cout<<("ReadFlowFile: could not open %s", filename);
		return;
	}
    
    int width, height;
    float tag;

    if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
	(int)fread(&width,  sizeof(int),   1, stream) != 1 ||
	(int)fread(&height, sizeof(int),   1, stream) != 1)
	{
		std::cout<<("ReadFlowFile: problem reading file %s", filename);
		return;
	}

    if (tag != TAG_FLOAT) // simple test for correct endian-ness
	{
		std::cout<<("ReadFlowFile(%s): wrong tag (possibly due to big-endian machine?)", filename);
		return;
	}

    // another sanity check to see that integers were read correctly (99999 should do the trick...)
    if (width < 1 || width > 99999)
	{
		std::cout<<("ReadFlowFile(%s): illegal width %d", filename, width);
		return;
	}

    if (height < 1 || height > 99999)
	{
		std::cout<<("ReadFlowFile(%s): illegal height %d", filename, height);
		return;
	}

    int nBands = 2;
    img = cv::Mat(height,width,CV_32FC2);

    //printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
    int n = nBands * width;
    for (int y = 0; y < height; y++) {
		float* ptr = (float*)(img.data + y*img.step.p[0]);
		if ((int)fread(ptr, sizeof(float), n, stream) != n)
		{
			std::cout<<("ReadFlowFile(%s): file is too short", filename);
			return;
		}
	}

    if (fgetc(stream) != EOF)
		std::cout<<("ReadFlowFile(%s): file is too long", filename);

    fclose(stream);
}

// write a 2-band image into flow file 
//void WriteFlowFile(cv::Mat& img , const char* filename)
//{
//    if (filename == NULL)
//	std::cout<<("WriteFlowFile: empty filename");
//
//    const char *dot = strrchr(filename, '.');
//    if (dot == NULL)
//	std::cout<<("WriteFlowFile: extension required in filename '%s'", filename);
//
//    if (strcmp(dot, ".flo") != 0)
//	std::cout<<("WriteFlowFile: filename '%s' should have extension '.flo'", filename);
//
//    CShape sh = img.Shape();
//    int width = sh.width, height = sh.height, nBands = sh.nBands;
//
//    if (nBands != 2)
//	std::cout<<("WriteFlowFile(%s): image must have 2 bands", filename);
//
//    FILE *stream = fopen(filename, "wb");
//    if (stream == 0)
//        std::cout<<("WriteFlowFile: could not open %s", filename);
//
//    // write the header
//    fprintf(stream, TAG_STRING);
//    if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
//	(int)fwrite(&height, sizeof(int),   1, stream) != 1)
//	std::cout<<("WriteFlowFile(%s): problem writing header", filename);
//
//    // write the rows
//    int n = nBands * width;
//    for (int y = 0; y < height; y++) {
//	float* ptr = &img.Pixel(0, y, 0);
//	if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
//	    std::cout<<("WriteFlowFile(%s): problem writing data", filename); 
//   }
//
//    fclose(stream);
//}


/*
int main() {

    try {
	CShape sh(5, 1, 2);
	CFloatImage img(sh);
	img.ClearPixels();
	img.Pixel(0, 0, 0) = -5.0f;
	char *filename = "test.flo";

	WriteFlowFile(img, filename);
	ReadFlowFile(img, filename);
    }
    catch (CError &err) {
	fprintf(stderr, err.message);
	fprintf(stderr, "\n");
	exit(1);
    }

    return 0;
}
*/