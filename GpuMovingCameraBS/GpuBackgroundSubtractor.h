#pragma once
#undef min
#undef max
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <thrust\device_vector.h>
#include "GpuSuperpixel.h"
#include "MRFOptimize.h"
#include "ASAPWarping.h"
#include "FlowComputer.h"
#include "SuperpixelComputer.h"
#include <fstream>
#include <curand_kernel.h>
//! defines the default value for BackgroundSubtractorLBSP::m_fRelLBSPThreshold
#define BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.333f)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD (3)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nMinColorDistThreshold
#define BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nBGSamples
#define BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES (50)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nRequiredBGSamples
#define BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nSamplesForMovingAvgs
#define BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS (25)
struct EdgePoint
{
	EdgePoint(int _x, int _y, float _theta,float _color):x(_x),y(_y),theta(_theta),color(_color)
	{}
	int x;
	int y;
	float color;//3��3�����ƽ����ɫ
	float theta;//�Ƕȣ�0~180
};
//row,colΪ���ĵ�size*size����ƽ����ɫ
float AvgColor(const cv::Mat& img, int row, int col, int size=3);
class GpuBackgroundSubtractor : public cv::BackgroundSubtractor {
public:
	//! full constructor
	GpuBackgroundSubtractor(float fRelLBSPThreshold=BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
									size_t nMinDescDistThreshold=BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD,
									size_t nMinColorDistThreshold=BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD,
									size_t nBGSamples=BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES,
									size_t nRequiredBGSamples=BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES,
									size_t nSamplesForMovingAvgs=BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS);
	//! default destructor
	virtual ~GpuBackgroundSubtractor();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! gpu Background subtraction operator
	virtual void GpuBSOperator(cv::InputArray image, cv::OutputArray fgmask);
	virtual void WarpInitialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	virtual void WarpBSOperator(cv::InputArray image, cv::OutputArray fgmask);
	//! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=0);
	//! unused, always returns nullptr
	virtual cv::AlgorithmInfo* info() const;
	//! returns a copy of the latest reconstructed background descriptors image
	virtual void getBackgroundImage(cv::OutputArray backgroundDescImage) const;
	//! returns the keypoints list used for descriptor extraction (note: by default, these are generated from the DenseFeatureDetector class, and the border points are removed)
	virtual std::vector<cv::KeyPoint> getBGKeyPoints() const;
	//! sets the keypoints to be used for descriptor extraction, effectively setting the BGModel ROI (note: this function will remove all border keypoints)
	virtual void setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints);
	virtual void refreshModel(float fSamplesRefreshFrac);
	virtual void warpRefreshModel(float fSamplesRefreshFrac);
	//! turns automatic model reset on or off
	void setAutomaticModelReset(bool);
	void upload(std::vector<cv::Point2f>& vec, cv::gpu::GpuMat& d_mat)
	{
		cv::Mat mat(1, vec.size(), CV_32FC2, (void*)&vec[0]);
		d_mat.upload(mat);
	}
	void download(const cv::gpu::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
 	{
 	    vec.resize(d_mat.cols);
 	    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
 	    d_mat.download(mat);
 	}
	void download(const cv::gpu::GpuMat& d_mat, std::vector<uchar>& vec)
 	{
 	    vec.resize(d_mat.cols);
 	    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
 	    d_mat.download(mat);
 	}
	void cloneModels();
	void GetTransformMatrix(const cv::Mat& gray, const cv::Mat pre_gray, cv::Mat& homoM, cv::Mat& affineM)
	{
		int max_count = 50000;	  // maximum number of features to detect
		double qlevel = 0.05;    // quality level for feature detection
		double minDist = 2;   // minimum distance between two feature points
		std::vector<uchar> status; // status of tracked features
		std::vector<float> err;    // error in tracking
		std::vector<cv::Point2f> features1,features2;  // detected features
		// detect the features
		cv::goodFeaturesToTrack(gray, // the image 
			features1,   // the output detected features
			max_count,  // the maximum number of features 
			qlevel,     // quality level
			minDist);   // min distance between two features

		// 2. track features
		cv::calcOpticalFlowPyrLK(gray, pre_gray, // 2 consecutive images
			features1, // input point position in first image
			features2, // output point postion in the second image
			status,    // tracking success
			err);      // tracking error

		int k=0;

		for( int i= 0; i < features1.size(); i++ ) 
		{

			// do we keep this point?
			if (status[i] == 1) 
			{

				//m_features.data[(int)m_points[0][i].x+(int)m_points[0][i].y*m_oImgSize.width] = 0xff;
				// keep this point in vector
				features1[k] = features1[i];
				features2[k++] = features2[i];
			}
		}
		features1.resize(k);
		features2.resize(k);

		std::vector<uchar> inliers(features1.size(),0);
		homoM= cv::findHomography(
			cv::Mat(features1), // corresponding
			cv::Mat(features2), // points
			inliers, // outputted inliers matches
			CV_RANSAC, // RANSAC method
			0.1); // max distance to reprojection point

		affineM = estimateRigidTransform(features1,features2,true);
	}
	void UpdateBackground(float* pfCurrLearningRate, int x, int y,size_t idx_ushrt, size_t idx_uchar, const ushort* nCurrIntraDesc, const uchar* nCurrColor);
	//����ģ��
	void UpdateModel(const cv::Mat& curImg, const cv::Mat& curMask);
	void WarpModels();
	void WarpImage(const cv::Mat img, cv::Mat& warpedImg);
	void getHomography(const cv::Mat& dImage, cv::Mat&  homography);
	void MotionEstimate(const cv::Mat& dImage, cv::Mat& homography);
	void ExtractEdgePoint(const cv::Mat& img, const cv::Mat& edge, cv::Mat& edgeThetaMat,std::vector<EdgePoint>& edgePoints);
	//��Ե��ƥ��
	void MapEdgePoint(const std::vector<EdgePoint>& ePoints1, const cv::Mat& edge2,const cv::Mat edgeThetamat, const const cv::Mat& transform, float deltaTheta, cv::Mat& matchMask);
	//��������ǰ�����˶��Ƿ��뱳��һ�£�ȥ������ǰ��
	void MaskHomographyTest(cv::Mat& mCurr, cv::Mat& curr, cv::Mat & prev, cv::Mat& homography, float* distance)
	{
		
		float threshold = 0.5;
		std::vector<cv::Point2f> currPoints, trackedPoints;
		std::vector<uchar> status; // status of tracked features
		std::vector<float> err;    // error in tracking
		for(int i=0; i<mCurr.cols; i++)
		{
			for(int j=0; j<mCurr.rows; j++)
				if(mCurr.data[i + j*mCurr.cols] == 0xff)
					currPoints.push_back(cv::Point2f(i,j));
		}
		if (currPoints.size() <=0)
			return;
		// 2. track features
		cv::calcOpticalFlowPyrLK(curr, prev, // 2 consecutive images
			currPoints, // input point position in first image
			trackedPoints, // output point postion in the second image
			status,    // tracking success
			err);      // tracking error

		// 2. loop over the tracked points to reject the undesirables
		int k=0;
		for( int i= 0; i < currPoints.size(); i++ ) {
			// do we keep this point?
			if (status[i] == 1) {
				// keep this point in vector
				currPoints[k] = currPoints[i];
				trackedPoints[k++] = trackedPoints[i];
			}
		}
		// eliminate unsuccesful points
		currPoints.resize(k);
		trackedPoints.resize(k);
		
		for(int i=0; i<k; i++)
		{
			cv::Point2f pt = currPoints[i];
			double* data = (double*)homography.data;
			float x = data[0]*pt.x + data[1]*pt.y + data[2];
			float y = data[3]*pt.x + data[4]*pt.y + data[5];
			float w = data[6]*pt.x + data[7]*pt.y + data[8];
			x /= w;
			y /= w;
			float d = abs(trackedPoints[i].x-x) + abs(trackedPoints[i].y - y);
			const size_t idx_char = (int)currPoints[i].x+(int)currPoints[i].y*mCurr.cols;
			//distance[idx_char]= d;
			if (d < threshold)
			{
								
				mCurr.data[idx_char] = 0x0;			
				
			}			
		}
	}
protected:
	//! background model keypoints used for LBSP descriptor extraction (specific to the input image size)
	std::vector<cv::KeyPoint> m_voKeyPoints;
	//! defines the current number of used keypoints (always tied to m_voKeyPoints)
	size_t m_nKeyPoints;
	//! input image size
	cv::Size m_oImgSize;
	//! input image channel size
	size_t m_nImgChannels;
	//! input image type
	int m_nImgType;
	//! indicates whether internal structures have already been initialized (LBSP lookup tables, samples, etc.)
	bool m_bInitializedInternalStructs;
	//! absolute minimal color distance threshold ('R' or 'radius' in the original ViBe paper, used as the default/initial 'R(x)' value here, paired with BackgroundSubtractorLBSP::m_nDescDistThreshold)
	const size_t m_nMinColorDistThreshold;
	//! number of different samples per pixel/block to be taken from input frames to build the background model (same as 'N' in ViBe/PBAS)
	const size_t m_nBGSamples;
	//! number of similar samples needed to consider the current pixel/block as 'background' (same as '#_min' in ViBe/PBAS)
	const size_t m_nRequiredBGSamples;
	//! number of samples to use to compute the learning rate of moving averages
	const size_t m_nSamplesForMovingAvgs;
	//! current frame index, frame count since last model reset & model reset cooldown counters
	size_t m_nFrameIndex, m_nFramesSinceLastReset, m_nModelResetCooldown;
	//! last calculated non-zero desc ratio
	float m_fLastNonZeroDescRatio;
	//! specifies whether automatic model reset is enabled or not
	bool m_bAutoModelResetEnabled;
	//! specifies whether Tmin/Tmax scaling is enabled or not
	bool m_bLearningRateScalingEnabled;
	//! current learning rate caps
	float m_fCurrLearningRateLowerCap, m_fCurrLearningRateUpperCap;
	//! current kernel size for median blur post-proc filtering
	int m_nMedianBlurKernelSize;
	//! specifies the px update spread range
	bool m_bUse3x3Spread;
	//! specifies the downsampled frame size (used for cam motion analysis)
	cv::Size m_oDownSampledFrameSize;
	//! absolute descriptor distance threshold
	const size_t m_nDescDistThreshold;
	//! LBSP internal threshold offset value -- used to reduce texture noise in dark regions
	const size_t m_nLBSPThresholdOffset;
	//! LBSP relative internal threshold (kept here since we don't keep an LBSP object)
	const float m_fRelLBSPThreshold;
	//! pre-allocated internal LBSP threshold values for all possible 8-bit intensity values
	size_t m_anLBSPThreshold_8bitLUT[UCHAR_MAX+1];
	size_t* d_anLBSPThreshold_8bitLUT;

	//! background model pixel color intensity samples (equivalent to 'B(x)' in PBAS, but also paired with BackgroundSubtractorLBSP::m_voBGDescSamples to create our complete model)
	std::vector<cv::Mat> m_voBGColorSamples;
	std::vector<cv::Mat> w_voBGColorSamples;
	cv::gpu::GpuMat d_voBGColorSamples,d_wvoBGColorSamples;
	//! background model descriptors samples (tied to m_voKeyPoints but shaped like the input frames)
	std::vector<cv::Mat> m_voBGDescSamples;
	std::vector<cv::Mat> w_voBGDescSamples;
	cv::gpu::GpuMat d_voBGDescSamples,d_wvoBGDescSamples;
	cv::gpu::GpuMat d_fModels, d_wfModels, d_bModels, d_wbModels;
	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	cv::Mat m_oUpdateRateFrame;
	cv::Mat w_oUpdateRateFrame;
	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	cv::Mat m_oDistThresholdFrame;
	cv::Mat w_oDistThresholdFrame;
	//! per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
	cv::Mat m_oVariationModulatorFrame;
	cv::Mat w_oVariationModulatorFrame;
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat m_oMeanLastDistFrame;
	cv::Mat w_oMeanLastDistFrame;
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	cv::Mat m_oMeanMinDistFrame_LT, m_oMeanMinDistFrame_ST;
	cv::Mat w_oMeanMinDistFrame_LT, w_oMeanMinDistFrame_ST;
	//! per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and control max learning rates globally)
	cv::Mat m_oMeanDownSampledLastDistFrame_LT, m_oMeanDownSampledLastDistFrame_ST;
	cv::Mat w_oMeanDownSampledLastDistFrame_LT, w_oMeanDownSampledLastDistFrame_ST;
	//! per-pixel mean raw segmentation results
	cv::Mat m_oMeanRawSegmResFrame_LT, m_oMeanRawSegmResFrame_ST;
	cv::Mat w_oMeanRawSegmResFrame_LT, w_oMeanRawSegmResFrame_ST;
	//! per-pixel mean final segmentation results
	cv::Mat m_oMeanFinalSegmResFrame_LT, m_oMeanFinalSegmResFrame_ST;
	cv::Mat w_oMeanFinalSegmResFrame_LT, w_oMeanFinalSegmResFrame_ST;
	//! a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	cv::Mat m_oUnstableRegionMask;
	cv::Mat w_oUnstableRegionMask;
	//! per-pixel blink detection results ('Z(x)')
	cv::Mat m_oBlinksFrame;
	cv::Mat w_oBlinksFrame;
	//! pre-allocated matrix used to downsample (1/8) the input frame when needed
	cv::Mat m_oDownSampledColorFrame;
	
	//! copy of previously used pixel intensities used to calculate 'D_last(x)'
	cv::Mat m_oLastColorFrame;
	cv::Mat w_oLastColorFrame;
	//! copy of previously used descriptors used to calculate 'D_last(x)'
	cv::Mat m_oLastDescFrame;
	cv::Mat w_oLastDescFrame;
	//! the foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	cv::Mat m_oRawFGMask_last;
	cv::gpu::GpuMat d_oFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc)
	cv::Mat m_oFGMask_last;
	cv::gpu::GpuMat d_FGMask_last;

	//! the input color frame
	cv::gpu::GpuMat d_CurrentColorFrame;
	//! output foreground mask
	cv::gpu::GpuMat d_FGMask;

	//������˶����³��ֵ����ػ�����ʧ�����ء�
	cv::gpu::GpuMat d_outMask;
	

	//! defines whether or not the subtractor is fully initialized
	bool m_bInitialized;
	size_t m_nPixels;
	//! pre-allocated CV_8UC1 matrices used to speed up morph ops
	cv::Mat m_oFGMask_PreFlood;
	cv::gpu::GpuMat d_oFGMask_PreFlood;
	cv::Mat m_oFGMask_FloodedHoles;
	cv::gpu::GpuMat d_oFGMask_FloodedHoles;
	cv::Mat m_oFGMask_last_dilated;
	cv::gpu::GpuMat d_oFGMask_last_dilated;
	cv::Mat m_oFGMask_last_dilated_inverted;
	cv::gpu::GpuMat d_oFGMask_last_dilated_inverted;
	cv::Mat m_oRawFGBlinkMask_curr;
	cv::gpu::GpuMat d_oRawFGBlinkMask_curr;
	cv::Mat m_oRawFGBlinkMask_last;
	cv::gpu::GpuMat d_oRawFGBlinkMask_last;

	cv::gpu::GoodFeaturesToTrackDetector_GPU* m_gpuDetector;
	cv::gpu::PyrLKOpticalFlow  d_pyrLk;
	cv::gpu::GpuMat d_gray;
	cv::gpu::GpuMat d_preGray;
	cv::gpu::GpuMat d_prevPts;
	cv::gpu::GpuMat d_currPts;
	cv::gpu::GpuMat d_status;
	cv::gpu::GpuMat d_bgMask;
	cv::Mat m_homography,m_invHomography;
	std::vector<uchar> m_status; // status of tracked features
	std::vector<cv::Point2f> m_points[2];
	double* d_homoPtr;
	//������˶����³��ֵ����ػ�����ʧ������
	uchar* d_outMaskPtr;
	uchar* m_outMaskPtr;
	size_t m_nOutPixels;
	cv::Mat m_preThetaMat,m_thetaMat;
	std::vector<EdgePoint> m_preEdgePoints, m_edgePoints;	
	cv::Mat m_preEdges,m_preGray,m_lastGray;
	cv::Mat m_edges,m_gray;	
	//���ڱ�Ե���ٵ������б�
	std::list<cv::Mat> m_grayList;
	std::list<cv::Mat> m_edgeList;
	std::list<cv::Mat> m_thetaMatList;
	std::list<std::vector<EdgePoint>> m_edgePointList;
	const size_t LIST_SIZE;
	//��Ե���پ���
	size_t  m_trackDist;
	//����������������
	cv::Mat m_features,m_preFeatures,m_mixFeatures,m_rawFGMask;
	

	cv::gpu::GpuMat d_features;
	GpuSuperpixel* m_gs;
	MRFOptimize* m_optimizer;
	curandState* d_randStates;
	std::ofstream m_ofstream;
	float* m_distance;

	cv::Mat m_warpedImg;
	ASAPWarping* m_ASAP;
	DenseOpticalFlowProvier* m_DOFP;
	cv::Mat m_flow,m_wflow;
	//����������������Ϊǰ���Ĵ�����������ĳ���޿��Ը���Ϊ����
	cv::Mat m_fgCounter;
	SuperpixelComputer * m_SPComputer;
	//������ƥ����
	std::vector<int> m_matchedId;
	cv::Mat m_img,m_preImg;
};
