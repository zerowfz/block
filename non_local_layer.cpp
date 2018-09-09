#include "caffe/layers/non_local_layer.hpp"


namespace caffe{

template <typename Dtype>
void LocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
  CHECK_EQ(bottom.size(),2)<<"nonlocal layer takes two blobs as input";
  CHECK_EQ(top.size(),2)<<"nonlocal layer takes two blobs as output";

  neibor_size_ = this->layer_param_.nonlocal_param().neibor_size();
  conv_out1_ = this->layer_param_.nonlocal_param().conv_out1();
  conv_out2_ = this->layer_param_.nonlocal_param().conv_out2();
  
  conv1_layer_.clear();
  conv2_layer_.clear();
  conv3_layer_.clear();

  CHECK_EQ(bottom[0]->num(),bottom[1]->num())<<"input has diffenrent batchsize";
  CHECK_EQ(bottom[0]->)
}

}
