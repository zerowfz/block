#include "caffe/layers/non_local_layer.hpp"


namespace caffe{

template <typename Dtype>
void NonLocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
  CHECK_EQ(bottom.size(),2)<<"number of input should be 2";
  CHECK_EQ(top.size(),2)<<"number of output should be 2";
  CHECK_EQ(bottom[0]->num(),bottom[1]->num())<<"input should has same batchsize";
  CHECK_EQ(bottom[0]->height(),bottom[1]->height())<<"input should has same height";
  CHECK_EQ(bottom[0]->width(),bottom[1]->width())<<"input should has same width";

  num_ = bottom[0]->num();
  fea_channel_ = bottom[0]->channel();
  corr_channel_ = bottom[1]->channel();// neibor size q*q
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  
  infea_dim_ = this->layer_param_.nonlocal_param().conv_output();


  LayerParameter conv_param;
  conv_param.mutable_convolution_param() = this->layer_param_.nonlocal_param().conv_param();
  conv1_layers_.reset(new ConvolutionLayer<Dtype>(conv_param));
  conv1_layers_->SetUp(bottom[0],conv1_top_);
  covn2_layers_.reset(new ConvolutionLayer<Dtype>(conv_param));
  conv2_layers_->SetUp(bottom[0],conv2_top_);
  conv_param.mutable_convolution_param()->set_num_output(fea_channel_);
  conv3_layers_.reset(new ConvolutionLayer<Dtype>(conv_param));
  conv3_layers_->SetUp(bottom[0],conv3_top_);

  //blob parameter
  if(!conv_param.mutable_convolution_param()->has_bias_term()){
    CHECK_EQ(this->blobs_.size(),3)<<"three param should be specific";
    this->blobs_[0].reset(conv1_layers_->blobs()[0]);
    this->blobs_[1].reset(conv2_layers_->blobs()[0]);
    this->blobs_[2].reset(conv3_layers_->blobs()[0]);
  }else {
    CHECK_EQ(this->blobs_.size(),6)<<"six param shold be specific";
    this->blobs_[0].reset(conv1_layers_->blobs()[0]);
    this->blobs_[1].reset(conv1_layers_->blobs()[1]);
    this->blobs_[2].reset(conv2_layers_->blobs()[0]);
    this->blobs_[3].reset(conv2_layers_->blobs()[1]);
    this->blobs_[4].reset(conv3_layers_->blobs()[0]);
    this->blobs_[5].reset(conv3_layers_->blobs()[1]);
  }

  //option for corresponence between pixel and its neighbor

}
template <typename Dtype>
void NonLocalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
    if(num_==bottom[0]->num() && )
}

}
