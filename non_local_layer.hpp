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
    virtual void emd_guassian()
}
}
