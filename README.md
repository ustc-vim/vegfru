# VegFru
The repository is for the paper "VegFru: A Domain-Specific Dataset for Fine-grained Visual Categorization".

Download Links: [[Paper]](http://home.ustc.edu.cn/~saihui/project/vegfru/iccv17_vegfru.pdf) [[Supplementary Material]](http://home.ustc.edu.cn/~saihui/project/vegfru/iccv17_sup_vegfru.pdf) [[Project Page]](http://vim.ustc.edu.cn/?product=vegfru) [[Download VegFru Dataset]](http://pan.baidu.com/s/1boSNcV9)
 
The VegFru dataset is available at [here](http://pan.baidu.com/s/1boSNcV9) and this code is for the implementation of HybridNet.
```
./caffe-hybridnet: modified from Caffe (https://github.com/BVLC/caffe)
./hybridnet-dataset: the prototxt defining the models for each dataset (FGVC-Aircrafts)
```

## Usage

```
cd caffe-hybridnet
sh data/jointaircrafts/model/v4_hyvggcpnet/ft_last_layer.sh
sh data/jointaircrafts/model/v4_hyvggcpnet/ft_all.sh
```
The pretrained models can be downloaded at [here](http://pan.baidu.com/s/1hrWGGSW).

## Citation
Please cite the following paper if you find this useful in your research:

	@InProceedings{Hou2017VegFru,
	  Title                    = {VegFru: A Domain-Specific Dataset for Fine-grained Visual Categorization},
	  Author                   = {Saihui Hou, Yushan Feng and Zilei Wang},
	  Booktitle                = {IEEE International Conference on Computer Vision},
	  Year                     = {2017}
	}
