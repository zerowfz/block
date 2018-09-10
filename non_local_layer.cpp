#include "caffe/layers/non_local_layer.hpp"


namespace caffe{

template <typename Dtype>
void LocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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


  LayerParameter conv_param1 = 
  LayerParameter conv_param2 =
  LayerParameter conv_param3 = 
  conv1_layers_.reset()
  conv1_layers->SetUp(bottom[0],conv1_top_);
  covn2_layers.reset()
  conv2_layers->SetUp(bottom[0],conv2_top_);
  conv3_layers.reset()
  conv3_layers->SetUp(bottom[0],conv3_top_);
}

}
