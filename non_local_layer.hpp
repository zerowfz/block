#ifndef CAFFE_NON_LOCAL_LAYER_HPP_
#define CAFFE_NON_LOCAL_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

/*
 * non local moudle in q*q neighbor*/

template <typename Dtype>
class NonLocalLayer:public Layer<Dtype>{
  public:
    explicit NonLocalLayer(const LayerParameter& param)
	    :Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		    const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		    const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {return "NonLocal";}
    virtual inline int ExactNumBottomBlobs() const {return 2;}
    virtual inline int ExactNumTopBlobs() const {return 2;}

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		    const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		    const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);
    // different operation 
    virtual void emd_guassian();
    int num_;
    int fea_channel_;
    int height_;
    int width_;
    int infea_dim_;
    
    shared_ptr<ConvolutionLayer<Dtype>> conv1_layers_;
    shared_ptr<ConvolutionLayer<Dtype>> conv2_layers_;
    shared_ptr<ConvolutionLayer<Dtype>> conv3_layers_;
    shared_ptr<SplitLayer<Dtype>> split_layers_;
    shared_ptr<PermuteLayer<Dtype>> permute_layers_;

    vector<Blob<Dtype>*> conv1_bottom_vec_;
    vector<Blob<Dtype>*> conv1_top_vec_;
    vector<Blob<Dtype>*> conv2_bottom_vec_;
    vector<Blob<Dtype>*> conv2_top_vec_;
    vector<Blob<Dtype>*> conv3_bottom_vec_;
    vector<Blob<Dtype>*> conv3_top_vec_;
    vector<Blob<Dtype>*> split_bottom_vec_;
    vector<Blob<Dtype>*> split_top_vec_;
    vector<Blob<Dtype>*> permute_layers_vec_;
    

}
}
