#include "caffe/layers/non_local_layer.hpp"


namespace caffe{

template <typename Dtype>
void NonLocalLayer<Dtype>::neibor2col(const Dtype* data_im,const int channel,const int height,const int width,
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
void NonLocalLayer<Dtype>::col2neibor(const Dtype* col,const int channel,const int height,const int width,
		const int kernel,const int center,Dtype* im){
    int center_h = center/width;
    int center_w = center%height;
    for(int kernel_row = 0;kernel_row<kernel;++kernel_row){
       int nei_h = center_h - kernel + kernel_row;
       nei_h = nei_h >= 0?nei_h:height+nei_h;
       nei_h = nei_h < height?nei_h:nei_h-height;
       for(int kernel_col = 0;kernel_col<kernel;++kernel_col)
          int nei_w = center_w - kernel +kernel_col;
          nei_w = nei_w >=0?nei_w:width+nei_w;
	  nei_w = nei_w<width?nei_w:nei_w-width;
	  for(int c=0;c<channel;++c){
	     im[(c*height+nei_h)*width+nei_w] += *(col++);
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
  kernel_ = sqrt(corr_channel_);
  
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

  top0_tem_vec_.push_back(new Blob<Dtype>());
  top0_tem_vec_[0].mutable_cpu_data()->Reshape(num_,height_,width_,fea_channel_);
  top1_tem_vec_.push_back(new Blob<Dtype>());
  top1_tem_vec_[0].mutable_cpu_data()->Reshape(num_,height_,width_,corr_channel_);
  LayerParamter permute_param;
  permute_param.mutable_permute_param()->add_order(0);
  permute_param.mutable_permute_param()->add_order(3);
  permute_param.mutable_permute_param()->add_order(1);
  permute_param.mutable_permute_param()->add_order(2);
  permute_layers0_.reset(new PermuteLayer<Dtype>(permute_param));
  permute_layers0_->Setup(top0_tem_vec_,vector<Blob<Dtype>*>{top[0]});

  LayerParamter permute_param;
  permute_param.mutable_permute_param()->add_order(0);
  permute_param.mutable_permute_param()->add_order(3);
  permute_param.mutable_permute_param()->add_order(1);
  permute_param.mutable_permute_param()->add_order(2);
  permute_layers1_.reset(new PermuteLayer<Dtype>(permute_param));
  permute_layers1_->Setup(top1_tem_vec_,vector<Blob<Dtype>*>{top[1]});

  //softmax 中间值；
  norm_cor_tem_.Reshape(num_,height_,width_,fea_channel_);
  //blob parameter
  //
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
  
  CHECK_EQ(top[0]->num(),num_);
  CHECK_EQ(top[0]->channel(),fea_channel_);
  CHECK_EQ(top[0]->height(),height_);
  CHECK_EQ(top[0]->width(),width_);
  CHECK_EQ(top[1]->channel(),corr_channel_);

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
    top0_tem_vec_[0]->mutable_cpu_data()->Reshape(num_,height_,width_,fea_channel_);
    top1_tem_vec_[1]->mutable_cpu_data()->Reshape(num_,height_,width_,corr_channel_);
    permute_layers0_->Reshape(top0_tem_vec_,vector<Blob<Dtype>*> {top[0]});
    permute_layers1_->Reshape(top1_tem_vec_,vector<Blob<Dtype>*> {top[1]});
    norm_cor_tem_.Reshape(num_,height_,width_,fea_channel_);
}

template <typename Dtype>
void NonLocalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
  Dtype* bottom_fea = bottom[0]->cpu_data();
  Dtype* bottom_cor = bottom[1]->cpu_data();
  Dtype* conv2_data = covn2_top_vec_[0]->cpu_data();
  Dtype* conv3_data = conv3_top_vec_[0]->cpu_data();
  Dtype* permute_data = permute_top_vec[0]->cpu_data();
  
  split_layer_->Forward_cpu(split_bottom_vec_,split_top_vec_);
  //3 convolution
  conv1_layers_->Forward_cpu(conv1_bottom_vec_,conv1_top_vec_);
  conv2_layers_->Forward_cpu(conv2_bottom_vec_,conv2_top_vec_);
  conv3_layers_->Forward_cpu(conv3_bottom_vec_,conv3_top_vec_);

  permute_layers_->Forward_cpu(conv1_top_vec_,permute_top_vec_);
  //对于conv1中的每一个特征，与conv2中的q*q个特征，做矩阵乘积运算  1*l ,l*q*q. 
  Blob<Dtype> qxq_l_fea;
  qxq_l_fea.Reshape(corr_channel_,infea_dim_,1,1);
  Blob<Dtype> mulit_sum;
  mulit_sum.Reshape(corr_channel_,1,1,1);
  caffe_set(corr_channel_,1,mulit_sum.mutable_cpu_data());
  Blob<Dtype> qxq_m_fea;
  qxq_m_fea.Reshape(corr_channel_,fea_channel_,1,1);
  Dtype* norm_data = norm_cor_tem_.mutable_cpu_data();
 
  Dtype* top_fea = top0_tem_vec_->mutable_cpu_data();
  Dtype* top_cor = top1_tem_vec_->mutable_cpu_data();
  for(int n=0;n<num_;++n){
      for(int j=0;j<height_*width_;++j){
          //得到conv2_data的q*q的矩阵
          neibor2col(conv2_data+conv2_top_vec_[0]->offset(n),infea_dim_,height_,
			  width_,kernel_,j,qxq_l_fea.->mutable_cpu_data());
	  int offset_each = top1_tem_vec_->offset(n,j/width_,j%width_);
	  //与conv1_data的1*l的特征求领域关系(l == infea_dim_)
	  caffe_cpu_gemv(CblasNoTrans,corr_channel_,infea_dim_,1,qxq_l_fea->cpu_data(),
			  permute_data+permute_top_vec_->offset(n,j/width_,j%width_),
			  0,top_cor+offset_each);
	  //计算该关系的softmax值

	  caffe_exp<Dtype>(corr_channel_,top_cor+offset_each,norm_data+offset_each);
	  Dtype* sum;
	  caffe_cpu_gemv(CblasNoTrans,1,corr_channel_,1,norm_data+offset_each,mulit_sum->cpu_data(),0,sum);        
	  for(int i=0;i<corr_channel_,++i){
	      *(norm_data+offset_each+i) /= (*sum); 
	  }
	  //得到conv3_data的q*q x m的值（m=fea_channel_）
          neibor2col(conv3_data+conv3_top_vec_[0]->offset(n),fea_channel_,height_,width_,
			  kernel_,j,qxq_m_fea->mutable_cpu_data());
	  caffe_cpu_gemv(CblasTrans,corr_channel_,fea_channel_,1,qxq_m_fea->cpu_data(),norm_data+offset_each,
			  0,top_fea+top0_tem_vec_->offset(n,j/width_,j%width_));
      }
  }
  permute_layers0_->Forward_cpu(top0_tem_vec_,vector<Blob*>{top[0]});
  permute_layers1_->Forward_cpu(top1_tem_vec_,vector<Blob*>{top[1]});
  caffe_add(num_*height_*width_*fea_channel_,top[0]->cpu_data(),split_top_vec_[3]->cpu_data(),
		  top[0]->mutable_cpu_data());
  caffe_add(num_*height_*width_*corr_channel_,top[1]->cpu_data(),bottom[1]->cpu_data(),
		  top[1]->mutable_cpu_data());
}

template <typename Dtype>
void NonLocalLayer::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){
    if(!propagate_down)return;
    Dtype* bottom_fea_diff = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_cor_diff = bottom[1]->mutable_cpu_diff();
    
    permute_layers0_->Backward_cpu(vector<Blob*>{top[0]},propagate_down,top0_tem_vec_);
    permute_layers1_->Backward_cpu(vector<Blob*>{top[1]},propagate_down,top1_tem_vec_);
    
    caffe_copy(bottom[0]->count(),top[0]->cpu_diff(),split_top_vec_[3]->mutable_cpu_diff());

    Dtype* top0_diff = top0_tem_vec_[0]->cpu_diff();
    Dtype* top1_diff = top1_tem_vec_[0]->mutable_cpu_diff();
    Dtype* top1_data = top1_tem_vec_[0]->cpu_data();
    Dtype* norm_data = norm_cor_tem_->cpu_data();
    Dtype* norm_diff = norm_cor_tem_->mutable_cpu_diff();
    Dtype* conv3_diff = conv3_top_vec_[0]->mutable_cpu_diff();
    caffe_set(conv3_top_vec_[0]->count(),0,conv3_diff);
    
    Blob<type> qxq_m;
    qxq_m.Reshape(corr_channel_,fea_channel_,1,1);
    Dtype* qxq_m_diff = qxq_m->mutable_cpu_diff();

    for(int n=0;n<num_;++n){
        for(int j=0;j<height_*width_;++j){
	    //计算conv3_data的diff
            int offset_cor = top1_data->offset(n,j/width_,j%width_);//corr 矩阵的offset（n,h,w,c）
	    int offset_fea = top0_data->offset(n,j/width_,j%width_);//fea矩阵的offset(n,h,w,c)
	    
            caffe_cpu_gemv(CblasTrans,1,fea_channel_,1,norm_data+offset_cor,top0_diff+offset_fea,0,qxq_m_diff);
	    col2neibor(qxq_m_diff,fea_channel_,height_,width_,kernel_,j,conv3_diff);
	    
	    //计算norm_data的导数
	    neibor2col(conv3_data,fea_channel_,height_,width_,kernel_,j,qxq_m_diff);
	    caffe_cpu_gemv(CblasNoTrans,cor_channel_,fea_channel_,1,qxq_m_diff,top0_diff+offset_fea,0,norm_diff+offset_cor);


	    

	    

	}
    }


}
}
