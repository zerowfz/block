#include "caffe/layers/non_local_layer.hpp"


namespace caffe{

template <typename Dtype>
void NonLocalLayer<Dtype>::neibor2col(const Dtype* data_im,const int channels,const int height,const int width,
const int kernel,const int center,Dtype* data_col){
    int center_h = center/width;
	int center_w = center%width;
	for(int kernel_row = 0;kernel_row<kernel;++kernel_row){
		int nei_h = center_h - kernel +kernel_row;
		nei_h = nei_h>=0?nei_h:height+nei_h;
		nei_h = nei_h<height?nei_h:nei_h-height;
	    for(int kernel_col = 0;kernel_col<kernel;++kernel_col){
			int nei_w = center_w - kernel + kernel_col;
			nei_w = nei_w>=0?nei_w:width+nei_w;
			nei_w = nei_w<width?nei_w:nei_w-width;
            for(int c = 0;c <channel;++c){
			    	*(data_col++) = data_im[(c*height+nei_h)*width+nei_w];
			}
		}
	}
}
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

  LayerParameter split_param;
  for(int i=0;i<4;++i)
	  split_top_vec_.push_back(new Blob<dtype>());
  split_layers_.reset(new SplitLayer<Dytpe>(split_param));
  split_bottom_vec_.push_back(bottom[0]);
  split_layers_->SetUp(split_bottom_vec_,split_top_vec_);

  LayerParameter conv_param;
  conv_param.mutable_convolution_param() = this->layer_param_.nonlocal_param().conv_param();
  conv1_layers_.reset(new ConvolutionLayer<Dtype>(conv_param));
  conv1_bottom_vec_.push_back(split_top_vec_[0]);
  conv1_layers_->SetUp(conv1_bottom_vec_,conv1_top_);

  covn2_layers_.reset(new ConvolutionLayer<Dtype>(conv_param));
  conv2_bottom_vec_.push_back(split_top_vec_[1]);
  conv2_layers_->SetUp(conv2_bottom_vec_,conv2_top_);

  conv_param.mutable_convolution_param()->set_num_output(fea_channel_);
  conv3_layers_.reset(new ConvolutionLayer<Dtype>(conv_param));
  conv3_bottom_vec_.push_back(split_top_vec_[2]);
  conv3_layers_->SetUp(conv3_bottom_vec_,conv3_top_);

  LayerParamter permute_param;
  permute_param.mutable_permute_param()->add_order(0);
  permute_param.mutable_permute_param()->add_order(2);
  permute_param.mutable_permute_param()->add_order(3);
  permute_param.mutable_permute_param()->add_order(1);
  permute_layers_.reset(new PermuteLayer<Dtype>(permute_param));
  permute_layers_->Setup(conv1_top_vec_,permute_top_vec_);
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
  
  top[0]->Reshape(num_,fea_channel_,height_,width_);
  top[1]->Reshape(num_,channel_channel_,height_,width_);
  //option for corresponence between pixel and its neighbor

}
template <typename Dtype>
void NonLocalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
    if(num_==bottom[0]->num() && height_ == bottom[0]->height() &&\
     width_==bottom[0]->width() && fea_channel_==bottom[0]->channel()\
     && corr_channel_==bottom[1]->channel())
	    return;
    num_ = bottom[0]->num();
    fea_channel_ = bottom[0]->channel();
    corr_channel_ = bottom[1]->channel();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    split_layers_->Reshape(split_bottom_vec_,split_top_vec_);
    conv1_layers_->Reshape(conv1_bottom_vec_,conv1_top_vec_);
    conv2_layers_->Reshape(conv2_bottom_vec_,conv2_top_vec_);
    conv3_layers_->Reshape(conv3_bottom_vec_,conv3_top_vec_);
    permute_layers_->Reshape(conv1_top_vec_,permute_top_vec_);
    top[0]->Reshape(num_,fea_channel_,height_,width_);
    top[1]->Reshape(num_,corr_channel_,height_,width_);
}

template <typename Dtype>
void NonLocalLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
  Dtype* bottom_fea = bottom[0]->cpu_data();
  Dtype* bottom_cor = bottom[1]->cpu_data();
  Dtype* top_fea = top[0]->mutable_cpu_data();
  Dtype* top_cor = top[1]->mutable_cpu_data();
  Dtype* conv2_data = covn2_top_vec_[0]->cpu_data();
  Dtype* conv3_data = conv3_top_vec_[0]->cpu_data();
  Dtype* permute_data = permute_top_vec[0]->cpu_data();
  
  split_layer_->Forward(split_bottom_vec_,split_top_vec_);
  //3 convolution
  conv1_layers_->Forward(conv1_bottom_vec_,conv1_top_vec_);
  conv2_layers_->Forward(conv2_bottom_vec_,conv2_top_vec_);
  conv3_layers_->Forward(conv3_bottom_vec_,conv3_top_vec_);

  permute_layers_->Forward(conv1_top_vec_,permute_top_vec_);
  //对于conv1中的每一个特征，与conv2中的q*q个特征，做矩阵乘积运算  1*l ,l*q*q. 
  for(int i=0;i<num_;++i){
      for(int j=0;j<height_*width_;++j){
      //得到conv2_data的q*q的矩阵

      }
  }
}

}
