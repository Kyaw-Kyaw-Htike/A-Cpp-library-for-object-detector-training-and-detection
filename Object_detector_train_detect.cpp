// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

#include "stdafx.h"

#include "fileIO_helpers.h"
#include "timer_ticToc.h"
//#include "typeExg_matlab_arma.h"
//#include "matlab_Eng_Wrapper_arma.h"
//#include "typeExg_opencv_arma.h"

#include "vl_feat_wrappers.h"
#include "hog_dollar_wrap.h"
#include "utils_KKH_opencv.h"
#include "utils_KKH_std_vector.h"
#include "matrix_class_KKH.h"

//#include "matlab_Eng_Wrapper_arma.h"

//#include "marray.hxx"

using namespace std; // for standard C++ lib

/*
Level 1 feature extraction. For mapping input images to
channel like feature "images" (color images are just a special case)
*/
class featL1_Base
{
public:
	virtual ~featL1_Base() {};
	// input should be of type CV_8UC1 or CV_8UC3; an image
	// output should be of a feature channel image CV_32FC(n)
	// where n can vary
	virtual cv::Mat extract(const cv::Mat &img) = 0;
	int get_shrinkage() { return shrinkage; }
	int get_nchannels() { return featNChannels; }
protected:
	int shrinkage; // must be 1, 4, 8, 12, etc
	int featNChannels; 
};

/*
Level 2 feature extraction.
From a patch corresponding to a channel image, it converts to
a feature vector. Simple flattening to make it a vector is just 
a special case of this.
*/
class featL2_Base
{
public:
	virtual ~featL2_Base() {};
	// input patchChannel should be of type CV_32FC(n) where n can vary
	// output of this function should be a row feature vector of type CV_32FC1
	virtual cv::Mat extract(const cv::Mat &patchChannel) = 0;
	int get_ndimsFeat() { return ndims_feat; }
protected:
	int ndims_feat;
};

/*
Classifies an input feature vector to give a score which denotes
the confidence of the classifier. The higher the score,
the more likely the sample belongs to the positive class.
*/
class classifier_Base
{
public:
	virtual ~classifier_Base() {};
	// input should be a row vector of type CV_32FC1.
	// output is a float number.
	virtual float classify(const cv::Mat &featVec) = 0;
	// train classifier
	virtual void train(const cv::Mat &featMatrix, const cv::Mat &labels) = 0;
protected:
	// if classification score > thresh, then +ve class, else -ve class.
	float thresh; 
};

class NMS_Base
{
public:
	virtual ~NMS_Base() {};
	virtual void suppress(const std::vector<cv::Rect> &dr,
		const std::vector<float> &ds, std::vector<cv::Rect> &dr_nms,
		std::vector<float> &ds_nms) = 0;
};

class featL1_naive : public featL1_Base
{
public:
	featL1_naive()
	{
		shrinkage = 1;
		featNChannels = 1;
	}
	cv::Mat extract(const cv::Mat &img) override
	{
		cv::Mat img_channels;
		cv::cvtColor(img, img_channels, CV_RGB2GRAY);
		img_channels.convertTo(img_channels, CV_32FC1);
		return img_channels;
	}
};

class hogVLFeatL1 : public featL1_Base
{
public:
	hogVLFeatL1(int nrows_img, int ncols_img, int nchannels_img) :
		hogObj(ncols_img, nrows_img, nchannels_img, 8, HOG_variant::HogVariantUoctti, 9, true)
	{
		featNChannels = hogObj.get_num_hogChannels(); // 31
		shrinkage = 8;
	}

	cv::Mat extract(const cv::Mat &img) override
	{
		cv::Mat img_temp;
		cv::cvtColor(img, img_temp, CV_RGB2BGR);
		img_temp.convertTo(img_temp, CV_32FC3);
		arma::Cube<float> img_arma;
		opencv2arma<float, 3>(img_temp, img_arma);
		cv::Mat feats_cv;
		vl_hog_w hogObj2(img_arma.n_cols, img_arma.n_rows, img_arma.n_slices);
		arma2opencv<float, 31>(hogObj2.extract_feat(img_arma), feats_cv);
		return feats_cv;
	}
private:
	vl_hog_w hogObj;
};

class lbpVLFeatL1 : public featL1_Base
{
public:
	lbpVLFeatL1() :
		lbpObj(false)
	{
		featNChannels = lbpObj.get_num_lbpChannels();
		shrinkage = 8;
	}

	cv::Mat extract(const cv::Mat &img) override
	{
		cv::Mat img_temp;
		cv::cvtColor(img, img_temp, CV_BGR2GRAY);
		img_temp.convertTo(img_temp, CV_32FC1);

		int nr = img.rows;
		int nc = img.cols;

		std::vector<float> img_vec(nr*nc);
		unsigned int cc = 0;	
		float* ptr_row;
		for (size_t i = 0; i < nr; i++)
		{
			ptr_row = img_temp.ptr<float>(i);
			for (size_t j = 0; j < nc; j++)
				img_vec[cc++] = ptr_row[j];
		}
		
		std::vector<float> H;
		int nr_H, nc_H, nch_H;
		lbpObj.extract_feat(img_vec.data(), nr, nc, H, nr_H, nc_H, nch_H, shrinkage);

		cv::Mat feats_cv(nr_H, nc_H, CV_32FC(nch_H));

		int ss[3] = { nr_H, nc_H, nch_H };
		cv::Mat feats_cv_temp = feats_cv.reshape(1, 3, ss);
		cc = 0;
		for (size_t k = 0; k < nch_H; k++)
			for (size_t j = 0; j < nc_H; j++)
				for (size_t i = 0; i < nr_H; i++)
					feats_cv_temp.at<float>(i, j, k) = H[cc++];

		return feats_cv;

	}
private:
	vl_lbp_w lbpObj;
};

class hogDollarFeatL1 : public featL1_Base
{
public:
	hogDollarFeatL1() = delete;
	hogDollarFeatL1(bool dalalHog, int shrinkage_ = 8) 
		:	hogObj()
	{
		if (!dalalHog) hogObj.set_params_falzen_HOG();
		featNChannels = hogObj.nchannels_hog();
		shrinkage = shrinkage_;
		hogObj.set_param_binSize(shrinkage);
	}

	cv::Mat extract(const cv::Mat &img) override
	{
		cv::Mat img_temp;
		cv::cvtColor(img, img_temp, CV_RGB2BGR);
		img_temp.convertTo(img_temp, CV_32FC3);

		int nr = img.rows; 
		int nc = img.cols; 
		int nch = img.channels();
		int nr_H = hogObj.nrows_hog(nr);
		int nc_H = hogObj.ncols_hog(nc);
		int nch_H = hogObj.nchannels_hog();

		std::vector<float> img_vec(nr*nc*nch);
		unsigned int cc = 0;
		for (size_t k = 0; k < nch; k++)
			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
					img_vec[cc++] = img_temp.at<cv::Vec<float, 3>>(i, j)[k];

		std::vector<float> H;
		hogObj.extract(img_vec.data(), nr, nc, nch, H);
		cv::Mat feats_cv(nr_H, nc_H, CV_32FC(nch_H));
		
		int ss[3] = { nr_H, nc_H, nch_H };
		cv::Mat feats_cv_temp = feats_cv.reshape(1, 3, ss);
		cc = 0;
		for (size_t k = 0; k < nch_H; k++)
			for (size_t j = 0; j < nc_H; j++)
				for (size_t i = 0; i < nr_H; i++)
					feats_cv_temp.at<float>(i, j, k) = H[cc++];
	
		return feats_cv;
	}
private:
	hog_dollar_wrap hogObj;
};

class hogLbpFeatL1 : public featL1_Base
{
public:
	hogLbpFeatL1()
		: hogObj()
	{
		hogObj.set_params_falzen_HOG();
		featNChannels = hogObj.nchannels_hog() + lbpObj.get_num_lbpChannels();
		shrinkage = 8;
	}
	
	cv::Mat extract(const cv::Mat &img) override
	{
		cv::Mat img_temp, img_gray, img_color;
		cv::cvtColor(img, img_temp, CV_RGB2BGR);
		img_temp.convertTo(img_color, CV_32FC3);
		cv::cvtColor(img, img_temp, CV_RGB2GRAY);
		img_temp.convertTo(img_gray, CV_32FC1);
		
		int nr = img.rows;
		int nc = img.cols;
		int nch = img.channels();

		std::vector<float> img_vec_color(nr*nc*nch);
		std::vector<float> img_vec_gray(nr*nc);

		int nr_hog, nc_hog, nch_hog, nr_lbp, nc_lbp, nch_lbp;
		unsigned int cc;

		cc = 0;
		for (size_t k = 0; k < nch; k++)
			for (size_t j = 0; j < nc; j++)
				for (size_t i = 0; i < nr; i++)
					img_vec_color[cc++] = img_color.at<cv::Vec<float, 3>>(i, j)[k];

		float* ptr_row;
		cc = 0;
		for (size_t i = 0; i < nr; i++)
		{
			ptr_row = img_gray.ptr<float>(i);
			for (size_t j = 0; j < nc; j++)
				img_vec_gray[cc++] = ptr_row[j];
		}

		std::vector<float> H_hog, H_lbp;

		// process HOG
		nr_hog = hogObj.nrows_hog(nr);
		nc_hog = hogObj.ncols_hog(nc);
		nch_hog = hogObj.nchannels_hog();
		hogObj.extract(img_vec_color.data(), nr, nc, nch, H_hog);

		// process LBP
		lbpObj.extract_feat(img_vec_gray.data(), nr, nc, H_lbp, nr_lbp, nc_lbp, nch_lbp, shrinkage);

		// just some checking
		if ((nr_hog != nr_lbp) || (nc_hog != nc_lbp))
		{
			printf("ERROR: no. of rows/cols of hog and lbp channel matrices do not match.\n");
			throw std::runtime_error("");
		}
		if (nch_hog + nch_lbp != featNChannels)
		{
			printf("ERROR: nch_hog + nch_lbp != featNChannels.\n");
			throw std::runtime_error("");
		}
						
		cv::Mat feats_cv(nr_hog, nc_hog, CV_32FC(featNChannels));

		int ss[3] = { nr_hog, nc_hog, featNChannels };
		cv::Mat feats_cv_temp = feats_cv.reshape(1, 3, ss);
		cc = 0;
		for (size_t k = 0; k < nch_hog; k++)
			for (size_t j = 0; j < nc_hog; j++)
				for (size_t i = 0; i < nr_hog; i++)
					feats_cv_temp.at<float>(i, j, k) = H_hog[cc++];
		cc = 0;
		for (size_t k = nch_hog; k < featNChannels; k++)
			for (size_t j = 0; j < nc_lbp; j++)
				for (size_t i = 0; i < nr_lbp; i++)
					feats_cv_temp.at<float>(i, j, k) = H_lbp[cc++];
		
		return feats_cv;
	}
private:
	hog_dollar_wrap hogObj;
	vl_lbp_w lbpObj;
};

class featL2_naive : public featL2_Base
{
public:
	featL2_naive(int ndims_feat_)
	{
		//ndims_feat = 128 * 64;
		//ndims_feat = 16 * 8 * 31;
		ndims_feat = ndims_feat_;
	}
	cv::Mat extract(const cv::Mat &patchChannel) override
	{
		//cout << "patchChannel num row, col & channels: " << patchChannel.rows << " " << patchChannel.cols << " " << patchChannel.channels() << endl;
		return patchChannel.clone().reshape(1, 1);		
	}
};

class classifier_naive : public classifier_Base
{
public:
	classifier_naive()
	{
		thresh = 0.5;
	}
	// input should be a row vector of type CV_32FC1.
	// output is a float number.
	float classify(const cv::Mat &featVec) override
	{
		return 1;
	}
	// train classifier
	void train(const cv::Mat &featMatrix, const cv::Mat &labels) override
	{
		return;
	}
};

class classifier_SVM_opencv : public classifier_Base
{
public:
	classifier_SVM_opencv()
	{
		thresh = 0;
	}
	// input should be a row vector of type CV_32FC1.
	// output is a float number.
	float classify(const cv::Mat &featVec) override
	{
		return w_lin.dot(featVec) + bias;
	}

	// train classifier
	void train(const cv::Mat &featMatrix, const cv::Mat &labels_) override
	{
		using namespace cv::ml;
		cv::Mat labels; labels_.convertTo(labels, CV_32SC1);
		Veck<int> mk(labels.total(),labels.ptr<int>(0), false);		
		// note: due to the fact that opencv treats label of -1 as the positive class, I need to
		// reverse it
		float classWeights[2] = { (mk == 1).size() / float(labels.total()), (mk == -1).size() / float(labels.total()) };
		cout << "class 0 weights = " << classWeights[0] << "; class 1 weights = " << classWeights[1] << endl;
		cv::Ptr<SVM> svm_obj = SVM::create();
		svm_obj->setType(SVM::Types::C_SVC);
		svm_obj->setC(0.01);
		svm_obj->setKernel(SVM::KernelTypes::LINEAR);
		svm_obj->setClassWeights(cv::Mat(1, 2, CV_32FC1, classWeights));

		cout << "featMatrix: " << featMatrix.rows << " " << featMatrix.cols << " " << featMatrix.channels() << endl;
		cout << "labels: " << labels.rows << " " << labels.cols << " " << labels.channels() << endl;
				
		cv::Ptr<TrainData> train_data = TrainData::create(featMatrix, ROW_SAMPLE, labels);	

		cout << "Training opencv SVM classifier...\n" << endl;

		svm_obj->train(train_data);

		//svm_obj->trainAuto(train_data, 10, 
		//	SVM::getDefaultGrid(SVM::C), SVM::getDefaultGrid(SVM::GAMMA),
		//	SVM::getDefaultGrid(SVM::P), SVM::getDefaultGrid(SVM::NU),
		//	SVM::getDefaultGrid(SVM::COEF), SVM::getDefaultGrid(SVM::DEGREE), true);
						
		cout << "Getting decision function...\n" << endl;
		cv::Mat alpha, svidx;
		double rho = svm_obj->getDecisionFunction(0, alpha, svidx);
		cv::Mat w_dec; w_dec = svm_obj->getSupportVectors(); // only one compressed vector
		w_lin = w_dec.reshape(1, 1).clone();
		bias = -rho;

		// to account for the fact that opencv treats labels with -1 as the positive class and
		// 1 as the negative class. After doing the following, during prediction, I can then
		// correctly perform dot product and add bias and if that result is > 0, then 
		// positive class, i.e. label = 1
		bias = -bias;
		w_lin = -w_lin;
	}

protected:
	cv::Mat w_lin;
	float bias;
};

class classifier_SVM_vlfeat : public classifier_Base
{
public:
	classifier_SVM_vlfeat()
	{
		thresh = 0;
	}
	// input should be a row vector of type CV_32FC1.
	// output is a float number.
	float classify(const cv::Mat &featVec) override
	{
		return w_lin.dot(featVec) + bias;
	}

	// load classifier model from file
	void load(const std::string &fpath)
	{
		cv::FileStorage fs(fpath, cv::FileStorage::READ);
		cv::Mat w;
		fs["w"] >> w; // assumes a column vector
		w_lin = w(cv::Range(0, w.rows-1), cv::Range(0, 1)).t();
		w_lin = w_lin.clone();
		bias = w.at<float>(w.rows - 1, 0);
		//cout << w_lin << endl;
		//cout << bias << endl;
		//cout << w_lin.rows << " " << w_lin.cols << endl;

		// to change the matlab's col majoring w to row major
		cv::Mat w_lin_CM(16, 8, CV_32FC(36));
		float *ptr = w_lin.ptr<float>(0);
		int cc = 0;
		for (size_t k = 0; k < 36; k++)
			for (size_t j = 0; j < 8; j++)
				for (size_t i = 0; i < 16; i++)
					w_lin_CM.at<cv::Vec<float, 36>>(i, j)[k] = ptr[cc++];
		
		w_lin = w_lin_CM.reshape(1, 1).clone();

	}

	// train classifier
	void train(const cv::Mat &featMatrix_, const cv::Mat &labels_) override
	{
		vl_svm_w svmObj;

		cv::Mat featMatrix;
		featMatrix_.convertTo(featMatrix, CV_64F);
		featMatrix = featMatrix.clone();

		cv::Mat labels;
		labels_.convertTo(labels, CV_64F);
		labels = labels.clone();

		printf("Training SVM classifier...\n");
		cout << "featMatrix: " << featMatrix.rows << " " << featMatrix.cols << " " << featMatrix.channels() << endl;
		cout << "labels: " << labels.rows << " " << labels.cols << " " << labels.channels() << endl;

		std::vector<double> lin_model = svmObj.train(featMatrix.ptr<double>(0), labels.ptr<double>(0),
			featMatrix.cols, featMatrix.rows, 0.001, true, true);

		printf("Classifier training completed.\n");
		w_lin.create(1, lin_model.size() - 1, CV_32FC1);
		std::vector<float> lin_model_float(lin_model.begin(), lin_model.end());
		std::copy(lin_model_float.begin(), lin_model_float.end() - 1, w_lin.ptr<float>(0));
		bias = lin_model_float[lin_model.size()-1];
	}

protected:
	cv::Mat w_lin;
	float bias;
};

class classifier_perceptron : public classifier_Base
{
public:
	classifier_perceptron()
	{
		thresh = 0;
	}
	// input should be a row vector of type CV_32FC1.
	// output is a float number.
	float classify(const cv::Mat &featVec) override
	{
		return w_lin.dot(featVec) + bias;
	}

	// train classifier
	void train(const cv::Mat &featMatrix, const cv::Mat &labels) override
	{		
		int nepochs = 1000;

		// build a Veck vector for the labels vector 
		Veck<int> labels_v(featMatrix.rows);
		std::copy(labels.ptr<int>(0), labels.ptr<int>(0) + featMatrix.rows, labels_v.begin());

		Veck<int> idx_pos = (labels_v == 1);
		Veck<int> idx_neg = (labels_v == -1);
		int npos = idx_pos.size();
		int nneg = idx_neg.size();

		idx_pos.shuffle_inplace();
		idx_neg.shuffle_inplace();
		
		int niters = std::max(npos, nneg);

		std::vector<cv::Mat> w_lin_all;
		std::vector<float> bias_all;
		std::vector<float> perf_all;
		w_lin_all.reserve(nepochs);
		bias_all.reserve(nepochs);
		perf_all.reserve(nepochs);

		// initialize linear weights and bias
		w_lin = cv::Mat::zeros(1, featMatrix.cols, CV_32FC1);
		bias = 0;

		int cc_pos = 0;
		int cc_neg = 0;
		bool pick_pos = true; // to alternate positive and negative data points
		int idx_picked;
		cv::Mat featVec;
		float label_groundtruth;
		
		for (size_t i = 0; i < nepochs; i++)
		{			
			int nwrongs = 0; // keep track of how many errors made in coming epoch
			cout << "Epoch = " << i << endl;
			for (size_t j = 0; j < niters; j++)
			{
				//cout << "Epoch " << i << ", Iter " << j << endl;
				// turn to pick a positive
				if (pick_pos)
				{
					//cout << "Turn to pick pos sample" << endl;
					// if have gone through all +ve examples, need to begin from the start
					// but after random shuffling
					if (cc_pos == npos)
					{
						//cout << "Entire pos set gone through. Restarting from beginning." << endl;
						cc_pos = 0;
						idx_pos.shuffle_inplace();
					}
					idx_picked = idx_pos[cc_pos];
					cc_pos++;
					pick_pos = false;
				}
				// turn to pick a positive
				else
				{
					//cout << "Turn to pick neg sample" << endl;
					// if have gone through all -ve examples, need to begin from the start
					// but after random shuffling
					if (cc_neg == nneg)
					{
						//cout << "Entire neg set gone through. Restarting from beginning." << endl;
						cc_neg = 0;
						idx_neg.shuffle_inplace();
					}
					idx_picked = idx_neg[cc_neg];
					cc_neg++;
					pick_pos = true;
				}			

				featVec = featMatrix.row(idx_picked);
				label_groundtruth = static_cast<float>(labels.at<int>(idx_picked, 0));

				//cout << "Picking training data num " << idx_picked << " which has label " << label_groundtruth << endl;

				// if wrong prediction, then update weight
				if (classify(featVec) * label_groundtruth <= 0)
				{					
					//cout << "Wrong classifier. Updating weights" << endl;
					w_lin += (label_groundtruth * featVec);
					bias += (label_groundtruth * 1.0);
					nwrongs++;
				}

			} //end j (iter)

			w_lin_all.push_back(w_lin);
			bias_all.push_back(bias);
			perf_all.push_back(static_cast<float>(niters - nwrongs)*100.0 / niters); // accuracy
			cout << "Training accuracy after this epoch = " << perf_all.back() << "%" << endl;
			//if (nwrongs == 0)
			if (nwrongs <= 10)
			{
				cout << "Early stopping due to nwrongs threshold." << endl;
				break; // Early stopping due to nwrongs threshold.
			}
		} // end i (epoch)
		
	}

protected:
	cv::Mat w_lin;
	float bias;
	//cv::Ptr<cv::ml::SVM> svm_obj;
};


class NMS_naive : public NMS_Base
{
public:
	void suppress(const std::vector<cv::Rect> &dr,
		const std::vector<float> &ds, std::vector<cv::Rect> &dr_nms,
		std::vector<float> &ds_nms) override
	{
		dr_nms = dr;
		ds_nms = ds;
	}
};

class NMSGreedy : public NMS_Base
{
	// C++ translation from the NMS matlab code of Tomasz Maliseiwicz who modified
	// Pedro Felzenszwalb's version to speed up. Tested and my C++ version and his matlab 
	// version give exactly
	// the same results. Out of 100 images, C++ version was found to be 1.78992 times 
	// faster than the matlab version on average and 1.74615 times faster on median.
	// This is quite good since the matlab version is already very heavily vectorized
	// and therefore very fast. 

public:
	NMSGreedy()
	{
		overlap_thresh = 0.5;
	}

	void set_thresh(float th)
	{
		overlap_thresh = th;
	}

	void suppress(const std::vector<cv::Rect> &dr,
		const std::vector<float> &ds, std::vector<cv::Rect> &dr_nms,
		std::vector<float> &ds_nms) override
	{
		arma::Mat<float> dr_(dr.size(), 4);
		arma::Col<float> ds_(dr.size());
		arma::Mat<float> dr_new;
		arma::Col<float> ds_new;

		for (size_t i = 0; i < dr.size(); i++)
		{
			dr_.at(i, 0) = dr[i].x;
			dr_.at(i, 1) = dr[i].y;
			dr_.at(i, 2) = dr[i].width;
			dr_.at(i, 3) = dr[i].height;
			ds_.at(i) = ds[i];
		}

		// process NMS
		merge_dets(dr_, ds_, dr_new, ds_new);

		int ndr_nms = dr_new.n_rows;
		dr_nms.resize(ndr_nms);
		ds_nms.resize(ndr_nms);

		for (size_t i = 0; i < ndr_nms; i++)
		{
			dr_nms[i].x = dr_new.at(i, 0);
			dr_nms[i].y = dr_new.at(i, 1);
			dr_nms[i].width = dr_new.at(i, 2);
			dr_nms[i].height = dr_new.at(i, 3);
			ds_nms[i] = ds_new.at(i);
		}
	}

	void merge_dets(const arma::Mat<float> &dr, const arma::Col<float> &ds,
		arma::Mat<float> &dr_new, arma::Col<float> &ds_new)
	{
		dr_new.set_size(0, 0);
		ds_new.set_size(0);

		if (dr.n_rows == 0) return;

		arma::Col<float> x1 = dr.col(0);
		arma::Col<float> y1 = dr.col(1);
		arma::Col<float> x2 = dr.col(0) + dr.col(2);
		arma::Col<float> y2 = dr.col(1) + dr.col(3);
		arma::Col<float> s = ds;

		arma::Col<float>area = (x2 - x1 + 1) % (y2 - y1 + 1);
		arma::uvec I = arma::sort_index(s);
		arma::Col<float> vals = s(I);	

		arma::uvec pick = arma::zeros<arma::uvec>(s.n_elem);
		unsigned int counter = 0;

		arma::Col<float> temp1, temp2, w, h, o, xx1, xx2, yy1, yy2;
		arma::uvec I_sub;

		while (I.n_elem > 0)
		{
			int last = I.n_elem;
			unsigned int i = I(last - 1);
			pick(counter) = i;
			counter++;

			if (last == 1) break;

			I_sub = I.subvec(0, last - 2);

			temp1 = x1(I_sub);
			temp2 = arma::repmat(arma::Mat<float>(&(x1(i)), 1, 1), temp1.n_elem, 1);
			xx1 = arma::max(temp2, temp1);

			temp1 = x2(I_sub);
			temp2 = arma::repmat(arma::Mat<float>(&(x2(i)), 1, 1), temp1.n_elem, 1);
			xx2 = arma::min(temp2, temp1);

			temp1 = y1(I_sub);
			temp2 = arma::repmat(arma::Mat<float>(&(y1(i)), 1, 1), temp1.n_elem, 1);
			yy1 = arma::max(temp2, temp1);

			temp1 = y2(I_sub);
			temp2 = arma::repmat(arma::Mat<float>(&(y2(i)), 1, 1), temp1.n_elem, 1);
			yy2 = arma::min(temp2, temp1);

			temp1 = xx2 - xx1 + 1;
			temp2 = arma::zeros<arma::Col<float>>(temp1.n_elem);
			w = arma::max(temp2, temp1);

			temp1 = yy2 - yy1 + 1;
			temp2 = arma::zeros<arma::Col<float>>(temp1.n_elem);
			h = arma::max(temp2, temp1);

			o = w % h / area(I_sub);
			I = I(arma::find(o <= overlap_thresh));

		}

		pick = pick.subvec(0, counter - 1);
		dr_new = dr.rows(pick);
		ds_new = ds(pick);
	}

private:

	float overlap_thresh;
};

class NMSOpencv : public NMS_Base
{

public:
	NMSOpencv()
	{
		use_meanshift = false;
	}

	NMSOpencv(bool use_meanshift_)
	{
		use_meanshift = use_meanshift_;
	}

	void suppress(const std::vector<cv::Rect> &dr,
		const std::vector<float> &ds, std::vector<cv::Rect> &dr_nms,
		std::vector<float> &ds_nms) override
	{
		if (!use_meanshift)
		{
			dr_nms = dr;
			cv::groupRectangles(dr_nms, 2);
			ds_nms.resize(dr_nms.size());
			std::fill(ds_nms.begin(), ds_nms.end(), 1);
		}

		else
		{
			dr_nms = dr;
			ds_nms = ds;
			//cv::groupRectangles_meanshift(dr_nms, ds_nms, scales, 0, cv::Size(64, 128));
		}

	}

private:
	bool use_meanshift;
};

/*
This class trains an object detector to detect objects using multi-scale sliding window scheme.
The feature extraction, classifier and NMS components are generic and can be anything as long
as they inherit from the abstract/base classes featL1_Base, featL2_Base, classifier_Base and 
NMS_Base and they override the respective methods.
Inputs to the training process are (1) directory to cropped positive patches 
(all same size as detection window size)
and (2) directory to negative images from where negatives and hard negatives will be data mined.
After the training process, I can run the detector on an image of any size to detect the object
category.
*/
class slidewin_detector
{
private:	

	// ===================================
	// Objects for feature extraction, classification and NMS
	// ===================================
	featL1_Base &featL1_obj;
	featL2_Base &featL2_obj;
	classifier_Base &classifier_obj;
	NMS_Base &NMS_Obj;

	// ===================================
	// for original image: the following params can be set by the user
	// in the constructor
	// ===================================
	// size of detection window ([0]: num rows; [1]: num cols)
	int winsize[2]; 
	// stride; same size for both horizontal and vertical slides
	int stride; 
	// the ratio between two scales for sliding window. 
	// the smaller, the finer the scales and thus, the more scales 
	// needed to processed
	double scaleratio;
	// in case user wants to limit the maximum number of scales
	// e.g. for very small objects in very large images.
	// normally, just set this number to a very large number (inf).
	int max_nscales; // if want to limit number of scales

	// ===================================
	// for feat channel "image": automatically computed params
	// during the constructor
	// ===================================
	// how much shrinking does the L1 feature extraction perform
	int shrinkage_channel; 
	// the stride on the channel "image". This should map back to the 
	// stride on the original image.
	int stride_channel;
	// size of window on the channel "image". This should map back to the 
	// winsize on the original image.
	int winsize_channel[2];
	// number of channels of the channel "image".
	int nchannels_channel;
	int ndims_feat; // length of the feature vector after L2 feature extraction

	// ===================================
	// output data of the private "process_img" method; these data correponds to
	// processing of a particular image size. These data members are written 
	// after the "process_img" method is called. The process_img is a private method
	// that does multi-scale sliding window processing including feature extraction,
	// classification, etc. The user can call public methods which make use of this
	// "process_img" method to access important functionalities.
	// ===================================
	// store info about for what image size the following "parameters" has been
	// prepared for.
	int nrows_img_prep, ncols_img_prep; 
	// no. of scales that image sliding window must process
	int num_scales;
	// vector of scales computed for sliding window; scales.size()==num_scales
	std::vector<double> scales;	
	// total no. of sliding windows for the image (across all the scales).
	unsigned int nslidewins_total;
	// vector of sliding window rectangles. dr.size()==nslidewins_total
	std::vector<cv::Rect> dr;
	// for each sliding window rectangle, which scale did it come from;
	// stores the index to std::vector<double>scales
	std::vector<unsigned int> idx2scale4dr;
	// vector of sliding window classification scores. ds.size()==nslidewins_total
	// ds will only be written if det_mode (which is one of the arguments to the
	// process_img method) is true.
	std::vector<float> ds;
	// matrix of features for randomly sampled sliding windows.
	// this will only be written if det_mode = false;
	cv::Mat feats_slidewin;

	// ===================================
	// Other data
	// ===================================
	// directory of cropped image patches for positive class for training classifier
	std::string dir_pos;
	// directory of full negative images for training classifier
	std::string dir_neg;
	// the file path to training data feature matrix and labels for training classifier
	std::string traindata_fpath_save;

	// ===================================
	// Private member functions
	// ===================================

	void check_params_constructor()
	{
		if (stride % shrinkage_channel != 0)
		{
			printf("ERROR: stride MOD shrinkage_channel != 0.\n");			
			throw std::runtime_error("");
		}			
		if (winsize[0] % shrinkage_channel != 0)
		{
			printf("ERROR: winsize[0] MOD shrinkage_channel != 0.\n");
			throw std::runtime_error("");
		}
		if (winsize[1] % shrinkage_channel != 0)
		{
			printf("ERROR: winsize[1] MOD shrinkage_channel != 0.\n");			
			throw std::runtime_error("");
		}		
	}

	// a convenient method for clearing all data in std::vector.
	// both vector size and capacity becomes zero.
	template <class T>
	void clean_vector(std::vector<T> &v)
	{
		//v.clear();
		std::vector<T>().swap(v);
	}

	// given an image, it will process in a sliding window manner
	// and write the "outputs"/results to certain private data members.
	// In the method argument, if save_feats == true, feature matrix of
	// of the sliding window rectangles will be saved in the data member
	// "feats_slidewin", where each row is the feature vector for one
	// sliding window rectangle (after L1 + L2 feature extraction).
	// be careful however that for very large images and very high dimensional
	// features, memory requirements might be too large.
	// if apply_classifier is false, the classifier will not be applied for the
	// sliding window. This should only be used when for example only want 
	// dr and feats_slidewin (when sampling features, dr, etc.)
	// dr will always be computed and saved in all cases.
	void process_img(const cv::Mat &img, bool save_feats=false, bool apply_classifier=true)
	{
		nrows_img_prep = img.rows; // to record the private data member
		ncols_img_prep = img.cols; // to record the private data member
		int nrows_img = nrows_img_prep; // for use locally in this method
		int ncols_img = ncols_img_prep; // for use locally in this method

		clean_vector(scales);
		clean_vector(dr);
		clean_vector(idx2scale4dr);
		if(apply_classifier) clean_vector(ds);
		
		// compute analytically how many scales there are for sliding window.
		// this formula gives the same answer as would be computed in a loop.
		num_scales = std::min(
			std::floor(std::log(static_cast<double>(nrows_img) / winsize[0]) / std::log(scaleratio)),
			std::floor(std::log(static_cast<double>(ncols_img) / winsize[1]) / std::log(scaleratio))) + 1;		

		// preallocate for efficiency
		scales.resize(num_scales);

		// find a tight upper bound on total no. of sliding windows needed
		double stride_scale, nsw_rows, nsw_cols;
		size_t nslidewins_total_ub = 0;
		for (size_t s = 0; s < num_scales; s++)
		{
			stride_scale = stride*std::pow(scaleratio, s);
			nsw_rows = std::floor(nrows_img / stride_scale) - std::floor(winsize[0] / stride) + 1;
			nsw_cols = std::floor(ncols_img / stride_scale) - std::floor(winsize[1] / stride) + 1;
			// Without the increment below, I get exact computation of number of sliding
			// windows, but just in case (to upper bound it)
			++nsw_rows; ++nsw_cols; 
			nslidewins_total_ub += (nsw_rows * nsw_cols);			
		}			

		//cout << "nrows_img: " << nrows_img << " " << "ncols_img: " << ncols_img << endl;
		//cout << "num_scales: " << num_scales << endl;
		//cout << "nslidewins_total_ub: " << nslidewins_total_ub << endl;
		//cout << "num channels of L1 feat channel image: " << nchannels_channel << endl;
		//cout << "ndims_feat: " << ndims_feat << endl;
		
		// preallocate/reserve for speed		
		dr.reserve(nslidewins_total_ub);
		idx2scale4dr.reserve(nslidewins_total_ub);
		if (apply_classifier) ds.reserve(nslidewins_total_ub);
		if (save_feats)
		{
			feats_slidewin = cv::Mat();
			feats_slidewin.reserve(nslidewins_total_ub);
		}

		// the resized image and the channel image
		cv::Mat img_cur, H, feat_vec;
		// reset counter for total number of sliding windows across all scales
		nslidewins_total = 0;

		for (size_t s = 0; s < num_scales; s++)
		{
			// compute how much I need to scale the original image for this current scale s
			scales[s] = std::pow(scaleratio, s);
			// get the resized version of the original image with the computed scale
			cv::resize(img, img_cur, cv::Size(), 1.0 / scales[s], 1.0 / scales[s], cv::INTER_LINEAR);

			// use L1 feature extractor to extract features from this resized image
			H = featL1_obj.extract(img_cur);

			// run sliding window in the channel image space 
			for (size_t i = 0; i < H.rows - winsize_channel[0] + 1; i += stride_channel)
			{
				for (size_t j = 0; j < H.cols - winsize_channel[1] + 1; j += stride_channel)
				{
					// save the current sliding window rectangle after mapping back:
					// (1) map from channel "image" space to image space (at this scale)
					// (2) map back to image space at this scale to original scale
										
					dr.push_back(cv::Rect(
						std::round(((j+1)*shrinkage_channel-shrinkage_channel)*scales[s]),
						std::round(((i+1)*shrinkage_channel - shrinkage_channel)*scales[s]),
						std::round((winsize[1]) * scales[s]),
						std::round((winsize[0]) * scales[s]))
					);

					// stores which scale of the original image this dr comes from
					idx2scale4dr.push_back(s);					
										
					// Get the channel image patch according to this current sliding window
					// rectangle, extract L2 features which will output a feature vector.
					feat_vec = featL2_obj.extract(
						H(
							cv::Rect(j, i, winsize_channel[1], winsize_channel[0])
						)
					);

					//cout << "feat_vec rows, cols and channels: " << feat_vec.rows << " " << feat_vec.cols << " " << feat_vec.channels() << endl;

					// apply classifier on the feature vector and save it
					if (apply_classifier) ds.push_back(classifier_obj.classify(feat_vec));

					// save the extracted features
					if (save_feats) feats_slidewin.push_back(feat_vec);
					
					++nslidewins_total;

				} // end j
			} //end i

		} //end s

		//cout << "nslidewins_total: " << nslidewins_total << endl;

	} //end method
	
public:

	// no default destructor allowed.
	slidewin_detector() = delete;

	// The constructor where the user need specify feature extraction, 
	// classifier and NMS objects. Default params are set for sliding window scheme.
	// if user wants to change these default sliding window params, 
	// use the method "set_params"
	slidewin_detector(featL1_Base &a, featL2_Base &b, classifier_Base &c, NMS_Base &d)
		:featL1_obj(a), featL2_obj(b), classifier_obj(c), NMS_Obj(d)
	{
		winsize[0] = 128; // num rows of detection window size 
		winsize[1] = 64; // num cols of detection window size
		stride = 8; 
		scaleratio = std::pow(2, 1 / 8.0);
		max_nscales = std::numeric_limits<int>::max();

		nrows_img_prep = 0;
		ncols_img_prep = 0;

		shrinkage_channel = featL1_obj.get_shrinkage();
		nchannels_channel = featL1_obj.get_nchannels();
		ndims_feat = featL2_obj.get_ndimsFeat();
		stride_channel = stride / shrinkage_channel;
		winsize_channel[0] = winsize[0] / shrinkage_channel;
		winsize_channel[1] = winsize[1] / shrinkage_channel;

		check_params_constructor();
	}

	// The constructor where the user need specify feature extraction, 
	// classifier and NMS objects, and also params for sliding window scheme.
	slidewin_detector(featL1_Base &a, featL2_Base &b, classifier_Base &c, NMS_Base &d,
		int winsize_nrows, int winsize_ncols, int stride_ = 8, 
		double scaleratio_ = std::pow(2, 1 / 8.0), 
		int max_nscales_ = std::numeric_limits<int>::max())
		:featL1_obj(a), featL2_obj(b), classifier_obj(c), NMS_Obj(d)
	{
		winsize[0] = winsize_nrows;
		winsize[1] = winsize_ncols;
		stride = stride_;
		scaleratio = scaleratio_;
		max_nscales = max_nscales_;

		nrows_img_prep = 0;
		ncols_img_prep = 0;
		
		
		shrinkage_channel = featL1_obj.get_shrinkage();
		nchannels_channel = featL1_obj.get_nchannels();
		ndims_feat = featL2_obj.get_ndimsFeat();
		stride_channel = stride / shrinkage_channel;
		winsize_channel[0] = winsize[0] / shrinkage_channel;
		winsize_channel[1] = winsize[1] / shrinkage_channel;	

		check_params_constructor();
	}

	
	// get feature vectors from given image from multi-scale sliding window space.
	// Useful for initially sampling negatives for training detector, etc.
	// if nsamples=-1, no sampling; return all features 
	// according to all sliding windows in order
	cv::Mat get_feats_img(const cv::Mat &img, int nsamples=-1)
	{
		// get all feats first
		process_img(img, true, false);

		if (nsamples < 0) return feats_slidewin;

		// prepare random number generator which will randomly
		// sample from the integer set {0,1,...,nslidewins_total-1}
		std::default_random_engine eng((std::random_device())());
		std::uniform_int_distribution<int> idis(0, nslidewins_total - 1);

		cv::Mat feats_sampled(nsamples, ndims_feat, CV_32FC1);

		for (size_t i = 0; i < nsamples; i++)
			feats_slidewin.row(idis(eng)).copyTo(feats_sampled.row(i));

		return feats_sampled;
	}
		
	// get rectangles from given image from multi-scale sliding window space.
	// if nsamples=-1, no sampling; return all sliding windows in order
	std::vector<cv::Rect> get_dr_img(const cv::Mat &img, int nsamples=-1)
	{
		// to get all dr first
		process_img(img, false, false);

		// if no sampling, then just return everything
		if (nsamples < 0) return dr; 

		// prepare random number generator which will randomly
		// sample from the integer set {0,1,...,nslidewins_total-1}
		std::default_random_engine eng((std::random_device())());
		std::uniform_int_distribution<int> idis(0, nslidewins_total-1);

		std::vector<cv::Rect> dr_sampled(nsamples);
		for (size_t i = 0; i < nsamples; i++)
			dr_sampled[i] = dr[idis(eng)];

		return dr_sampled;
	}

	// detect objects on the given image with the classifier
	void detect(const cv::Mat &img, std::vector<cv::Rect> &dr_, std::vector<float> &ds_, float dec_thresh=0, bool apply_NMS=true)
	{
		process_img(img, false, true);
		std::vector<cv::Rect> dr_temp;
		std::vector<float> ds_temp;
		if (apply_NMS)
		{
			std::vector<cv::Rect> dr_nms;
			std::vector<float> ds_nms;
			NMS_Obj.suppress(dr, ds, dr_nms, ds_nms);
			dr_temp = dr_nms;
			ds_temp = ds_nms;
		}
		else
		{
			dr_temp = dr;
			ds_temp = ds;
		}		

		int nrects = dr_temp.size();
		dr_.clear(); 
		ds_.clear();
		dr_.reserve(nrects);
		ds_.reserve(nrects);

		for (size_t i = 0; i < nrects; i++)
		{
			if (ds_temp[i] > dec_thresh)
			{
				dr_.push_back(dr_temp[i]);
				ds_.push_back(ds_temp[i]);
			}
		}

	}
	
	// train by extracting features from a directory of cropped positive patches, a directory of
	// full negative images (where hard negs will be mined). Optionally, all the extracted features
	// can be saved so that later on, if desired, I can use other overloaded train function
	// which just loads the saved features and labels for training
	void train(std::string dir_pos_, std::string dir_neg_, bool save_train_feats=false, std::string traindata_fpath_save="")
	{
		// just for recording so that in the future, I have a record of which training data
		// the detector was trained with
		dir_pos = dir_pos_; 
		dir_neg = dir_neg_;

		// read in image full path names
		std::vector<std::string> fnames_pos, fnames_neg;

		dir_fnames(dir_pos, { "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff" }, fnames_pos);
		dir_fnames(dir_neg, { "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff" }, fnames_neg);
		int npos = fnames_pos.size();
		int nnegImg = fnames_neg.size();

		// read cropped patches to form positive class of the dataset
		cv::Mat feats_pos(npos, ndims_feat, CV_32FC1);
		cv::Mat img;
		printf("Extracting features from cropped +ve class...\n");
		for (size_t i = 0; i < npos; i++)
		{
			img = cv::imread(fnames_pos[i]);
			get_feats_img(img).copyTo(feats_pos.row(i));
		}
		printf("Extracting +ve features done.\n");
		cout << "feats_pos info: " << feats_pos.rows << " " << feats_pos.cols << " " << feats_pos.channels() << endl;
		
		// random sample negative patches and features from negative images
		int num_ini_negImg = 100;
		int num_nsamples_per_img = 100;
		vector<cv::Mat> feats_neg_ini_vec(num_ini_negImg);
		for (size_t i = 0; i < num_ini_negImg; i++)
		{
			img = cv::imread(fnames_neg[i]);
			feats_neg_ini_vec[i] = get_feats_img(img, num_nsamples_per_img);
		}
		cv::Mat feats_neg_ini;
		cv::vconcat(feats_neg_ini_vec, feats_neg_ini);
		cout << "feats_neg_ini info: " << feats_neg_ini.rows << " " << feats_neg_ini.cols << " " << feats_neg_ini.channels() << endl;
		
		// train classifier with current initially collected dataset
		cv::Mat labels(npos + feats_neg_ini.rows, 1, CV_32SC1);
		labels(cv::Range(0, npos), cv::Range(0,1)).setTo(1);
		labels(cv::Range(npos, npos + feats_neg_ini.rows), cv::Range(0, 1)).setTo(-1);
		cv::Mat feats_train;
		cv::Mat feats_train_[2] = { feats_pos, feats_neg_ini };
		cv::vconcat(feats_train_, 2, feats_train);
		classifier_obj.train(feats_train, labels);		
		
		// go through negative images to find hard negs
		cout << "Looking for hard negs...\n" << endl;
		int nhardnegs = 10000;
		std::vector<cv::Rect> dr_dets;
		std::vector<float> ds_dets;
		unsigned int nfp, tnfp;
		cv::Mat feats_neg_hard;
		cv::Mat img_roi;
		feats_neg_hard.reserve(nhardnegs);

		tnfp = 0;
		for (size_t i = 0; i < nnegImg; i++)
		{
			img = cv::imread(fnames_neg[i]);
			detect(img, dr_dets, ds_dets, 0, true);
			nfp = dr_dets.size();

			for (size_t j = 0; j < nfp; j++)
			{	
				//cv::rectangle(img, dr_dets[j], cv::Scalar(255, 0, 0, 0), 2);
				// just in case the given dr_deets[j] overshoots a bit (by 1 or 2 pixels)
				// the image boundary, in order to prevent crashing
				if (dr_dets[j].x + dr_dets[j].width >= img.cols || dr_dets[j].y + dr_dets[j].height >= img.rows)
					continue;
				img_roi = img(dr_dets[j]);
				feats_neg_hard.push_back(get_feats_img(img_roi, 1));
				tnfp++;
			}
			//cv::imshow("win", img); cv::waitKey(0);
			
			cout << "Number of false +ves found in image " << i << " = " << nfp << endl;
			cout << "Total Number of false +ves collected so far = " << tnfp << endl;

			if (tnfp >= nhardnegs)
			{
				cout << "Stopped due to having reached target of " << nhardnegs << " hard neg samples" << endl;
				break;
			}
		}

		// retrain classifier
		cout << "Retraining classifier with hard negs...\n" << endl;
		feats_train_[0] = feats_train;
		feats_train_[1] = feats_neg_hard;
		cv::vconcat(feats_train_, 2, feats_train);
		cv::Mat labels_hardneg(tnfp, 1, CV_32SC1);
		labels_hardneg.setTo(-1);
		cv::Mat labels_[2] = { labels, labels_hardneg };
		cv::vconcat(labels_, 2, labels);

		// optionally, save the features (i.e. pos, neg and hard neg features & all labels)
		if (save_train_feats)
		{
			cout << "Saving feature matrix and labels...\n" << endl;
			cv::FileStorage fs(traindata_fpath_save, cv::FileStorage::WRITE);
			fs << "feats_train" << feats_train << "labels" << labels;
			fs.release();
		}

		cout << "Training classifier...\n" << endl;
		classifier_obj.train(feats_train, labels);
	}

	// train by loading feature matrix and labels from the saved file
	void train(std::string traindata_fpath_save_)
	{
		traindata_fpath_save = traindata_fpath_save_; // just record it
		cout << "Loading feature matrix and labels...\n" << endl;
		cv::FileStorage fs(traindata_fpath_save, cv::FileStorage::READ);
		cv::Mat feats_train, labels;
		fs["feats_train"] >> feats_train;
		fs["labels"] >> labels;
		fs.release();

		cout << "Training classifier...\n" << endl;
		classifier_obj.train(feats_train, labels);
	}

};






int main(int argc, char* argv[])
{	
		
	//cv::Mat img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00001.png");
	cv::Mat img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00000.png");
	//featL1_naive featL1_obj;
	//hogVLFeatL1 featL1_obj(128, 64, 3);
	//lbpVLFeatL1 featL1_obj(false);
	//hogLbpFeatL1 featL1_obj;
	//hogDollarFeatL1 featL1_obj(false);
	hogDollarFeatL1 featL1_obj(false, 8);
	//featL2_naive featL2_obj(16*8*featL1_obj.get_nchannels());
	//featL2_naive featL2_obj(16 * 8 * 31);
	featL2_naive featL2_obj(16 * 8 * featL1_obj.get_nchannels());
	//featL2_naive featL2_obj(4 * 4 * featL1_obj.get_nchannels());
	//featL2_naive featL2_obj(128*64);

	//classifier_naive classifier_obj;
	//classifier_SVM_vlfeat classifier_obj;
	classifier_perceptron classifier_obj;
	//classifier_SVM_opencv classifier_obj;
	//classifier_obj.load("w_saved.xml");
	//NMS_naive nms_obj;
	NMSGreedy nms_obj;
	//NMSOpencv nms_obj;
	slidewin_detector s(featL1_obj, featL2_obj, classifier_obj, nms_obj, 128, 64, 8);
	//slidewin_detector s(featL1_obj, featL2_obj, classifier_obj, nms_obj, 16, 16, 4);

	std::string dir_pos = "D:/Research/Datasets/INRIAPerson_Piotr/Train/imgs_crop_context/";
	//std::string dir_pos = "D:/Research/Datasets/INRIAPerson_Piotr/Train/16x16_outline_patches_from_imgs_crop_context/";
	std::string dir_neg = "D:/Research/Datasets/INRIAPerson_Piotr/Train/images/set00/V001/";
	std::string fpath_save = "C:/Users/Kyaw/Desktop/train_imgs_pos/trainData_16x16patches_HOG.xml";
	//s.train(dir_pos, dir_neg, true, fpath_save);
	s.train(dir_pos, dir_neg);
	//s.train(fpath_save);

	std::vector<cv::Rect> dr; std::vector<float> ds;
	timer_ticToc tt;
	tt.tic();
	s.detect(img, dr, ds, true);
	cout << "Time taken = " << tt.toc() << " secs" << endl;
	for (size_t i = 0; i < dr.size(); i++)
	{
		if (ds[i] > 0)
			cv::rectangle(img, dr[i], cv::Scalar(255, 0, 0, 0), 2);
	}
	imshow("win", img);
	cv::waitKey(0);
	//
	//img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00001.png");
	//s.detect(img, dr, ds, true);
	//for (size_t i = 0; i < dr.size(); i++)
	//{
	//	if (ds[i] > 0)
	//	{
	//		cv::rectangle(img, dr[i], cv::Scalar(255, 0, 0, 0), 2);
	//		//cout << ds[i] << endl;
	//	}

	//}
	//imshow("win", img);
	//cv::waitKey(0);


	
	
	//cout << m2.rows << " " << m2.cols << " " << m2.channels() << endl;
	
	//timer_ticToc tt;
	//tt.tic();	
	////cv::Mat feats_sampled = s.get_feats_img(img, 100);
	//img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Train/imgs_crop_context/INRIA_pos_00001.png");
	//std::vector<cv::Rect> dr_sampled = s.get_dr_img(img, -1);
	//cv::Mat feats_sampled = s.get_feats_img(img, -1);
	//cout << "Time taken = " << tt.toc() << " secs" << endl;

	//for (size_t i = 0; i < dr_sampled.size(); i++)
	//{
	//	cv::Mat img2 = img.clone();
	//	cv::rectangle(img2, dr_sampled[i], cv::Scalar(255, 0, 0, 0), 2);
	//	imshow("win", img2);
	//	cv::waitKey(0);
	//}

	//cout << "feats_sampled " << feats_sampled.rows << " " << feats_sampled.cols << endl;
	//cout << feats_sampled << endl;
	//for (size_t i = 0; i < feats_sampled.rows; i++)
	//{
	//	cv::Mat temp;
	//	feats_sampled.row(i).reshape(1, 128).convertTo(temp, CV_8UC1);
	//	imshow("win", temp);
	//	cv::waitKey(0);
	//}

	//cv::imshow("win", img); cv::waitKey(0);
	//cv::Rect roi(10, 10, 64, 128);
	//cv::Mat img2 = cv::Mat(img, roi);
	//cout << img2.rows << " " << img2.cols << endl;
	//cv::imshow("win", img2); cv::waitKey(0);

	//slidewin_detector_cv s;

	//slidewin_detector s;
	//s.strideX = 8;
	//s.strideY = 8;

	//timer_ticToc tt;
	//tt.tic();
	//for (size_t i = 0; i < 1; i++)
	//{
	//	s.extract_patches(img);
	//}	
	//double time_taken = tt.toc();
	//cout << "Time for 1 frames = " << time_taken << " secs" << endl;
	//cout << "Time for 30 frame = " << time_taken * 30 << " secs" << endl;	


	return 0;
}
